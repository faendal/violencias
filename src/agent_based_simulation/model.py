import mesa
import random
import pandas as pd
from .agents import ExpedienteAgente, DependenciaAgente


class AuditoriaOperativaModel(mesa.Model):
    def __init__(
        self, df_prophet, sim_params, matriz_markov, capacidades, pct_fin_semana
    ):
        super().__init__()
        self.df_prophet = df_prophet.reset_index(drop=True)
        self.df_prophet["Fecha"] = pd.to_datetime(
            self.df_prophet["Fecha"]
        )
        self.sim_params = sim_params
        self.matriz_markov = matriz_markov
        self.pct_fin_semana = pct_fin_semana

        self.current_step = 0
        self.case_id_counter = 0
        self.schedule = mesa.time.RandomActivation(self)

        self.dependencias = {}
        for servicio, cap in capacidades.items():
            dep = DependenciaAgente(f"Dep_{servicio}", self, servicio, cap)
            self.dependencias[servicio] = dep
            self.schedule.add(dep)

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
        if self.current_step < len(self.df_prophet):
            self.nuevos_casos_hoy = int(
                round(
                    max(
                        0, self.df_prophet.loc[self.current_step, "Casos Pronosticados"]
                    )
                )
            )
            fecha_actual = self.df_prophet.loc[self.current_step, "Fecha"]
            es_fin_semana = fecha_actual.weekday() >= 5  # 5=Sábado, 6=Domingo
        else:
            self.nuevos_casos_hoy = 0
            es_fin_semana = False

        nats = list(self.sim_params["dist_naturaleza"].keys())
        p_nats = list(self.sim_params["dist_naturaleza"].values())
        edades = list(self.sim_params["dist_edad"].keys())
        p_edades = list(self.sim_params["dist_edad"].values())

        for _ in range(self.nuevos_casos_hoy):
            naturaleza = random.choices(nats, weights=p_nats)[0] if nats else "Sin Dato"
            edad = random.choices(edades, weights=p_edades)[0] if edades else "Sin Dato"

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
                edad=edad,
                naturaleza=naturaleza,
                ruta_requerida=ruta_requerida,
            )
            self.schedule.add(nuevo_expediente)
            self.case_id_counter += 1

        for servicio, dependencia in self.dependencias.items():
            cola_servicio = [
                ag
                for ag in self.schedule.agents
                if isinstance(ag, ExpedienteAgente)
                and ag.estado_ruta.get(servicio) == "Pendiente"
            ]

            cola_servicio.sort(key=lambda x: (x.prioridad, -x.dias_en_espera_total))

            dependencia.procesar_cola(cola_servicio, es_fin_semana, self.pct_fin_semana)

        self.schedule.step()
        self.datacollector.collect(self)
        self.current_step += 1
