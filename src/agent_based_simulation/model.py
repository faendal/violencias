import mesa
import random
from .agents import ExpedienteAgente, DependenciaAgente


class AuditoriaOperativaModel(mesa.Model):
    def __init__(self, df_prophet, sim_params, matriz_markov, capacidades):
        super().__init__()
        self.df_prophet = df_prophet.reset_index(drop=True)
        self.sim_params = sim_params
        self.matriz_markov = (
            matriz_markov  # Probabilidades de SIVIGILA (Ej: Física -> 57% Protección)
        )

        self.current_step = 0
        self.case_id_counter = 0
        self.schedule = mesa.time.RandomActivation(self)

        # 1. Instanciar las Dependencias Institucionales
        self.dependencias = {}
        for servicio, cap in capacidades.items():
            dep = DependenciaAgente(f"Dep_{servicio}", self, servicio, cap)
            self.dependencias[servicio] = dep
            self.schedule.add(dep)

        # 2. Recolector de Métricas Operativas Duras
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Nuevos Casos (Prophet)": lambda m: getattr(m, "nuevos_casos_hoy", 0),
                "Casos Completados": lambda m: sum(
                    [
                        1
                        for a in m.schedule.agents
                        if isinstance(a, ExpedienteAgente) and a.esta_completado()
                    ]
                ),
                "Backlog Total": lambda m: sum(
                    [
                        1
                        for a in m.schedule.agents
                        if isinstance(a, ExpedienteAgente) and not a.esta_completado()
                    ]
                ),
                "Backlog Salud Mental": lambda m: m.contar_pendientes("Salud Mental"),
                "Backlog Proteccion": lambda m: m.contar_pendientes("Proteccion"),
                "Fallo Administrativo (>30 días)": lambda m: sum(
                    [
                        1
                        for a in m.schedule.agents
                        if isinstance(a, ExpedienteAgente) and a.fallo_administrativo
                    ]
                ),
            }
        )

    def contar_pendientes(self, servicio):
        return sum(
            [
                1
                for a in self.schedule.agents
                if isinstance(a, ExpedienteAgente)
                and a.estado_ruta.get(servicio) == "Pendiente"
            ]
        )

    def step(self):
        # 1. Ingesta de la Demanda Diaria (Señal de Prophet)
        if self.current_step < len(self.df_prophet):
            self.nuevos_casos_hoy = int(
                round(
                    max(
                        0, self.df_prophet.loc[self.current_step, "Casos Pronosticados"]
                    )
                )
            )
        else:
            self.nuevos_casos_hoy = 0

        # 2. Generación de Expedientes (Sembrado Empírico)
        nats = list(self.sim_params["dist_naturaleza"].keys())
        p_nats = list(self.sim_params["dist_naturaleza"].values())

        for _ in range(self.nuevos_casos_hoy):
            naturaleza = random.choices(nats, weights=p_nats)[0] if nats else "Sin Dato"

            # 3. Enrutamiento Markoviano (Sorteo basado en datos históricos SIVIGILA)
            probs = self.matriz_markov.get(
                naturaleza, {"Salud Mental": 0.3, "Proteccion": 0.5}
            )
            ruta_requerida = {
                "Salud Mental": random.random() < probs.get("Salud Mental", 0),
                "Proteccion": random.random() < probs.get("Proteccion", 0),
            }

            nuevo_expediente = ExpedienteAgente(
                f"Exp_{self.case_id_counter}",
                self,
                estrato="N/A",
                edad="N/A",
                naturaleza=naturaleza,
                ruta_requerida=ruta_requerida,
            )
            self.schedule.add(nuevo_expediente)
            self.case_id_counter += 1

        # 4. Procesamiento de Colas en Dependencias (Resolución del cuello de botella)
        for servicio, dependencia in self.dependencias.items():
            # Construir la cola de expedientes que esperan ESTE servicio en específico
            cola_servicio = [
                ag
                for ag in self.schedule.agents
                if isinstance(ag, ExpedienteAgente)
                and ag.estado_ruta.get(servicio) == "Pendiente"
            ]
            # Ordenar por antigüedad (Prioridad FIFO: First In, First Out)
            cola_servicio.sort(key=lambda x: x.dias_en_espera_total, reverse=True)

            # La dependencia atiende hasta donde le alcanza la capacidad
            dependencia.procesar_cola(cola_servicio)

        self.schedule.step()
        self.datacollector.collect(self)
        self.current_step += 1
