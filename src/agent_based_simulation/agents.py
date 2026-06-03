import mesa


class ExpedienteAgente(mesa.Agent):
    def __init__(self, unique_id, model, estrato, edad, naturaleza, ruta_requerida):
        super().__init__(unique_id, model)
        self.estrato = estrato
        self.edad = edad
        self.naturaleza = naturaleza
        self.ruta_requerida = ruta_requerida

        self.estado_ruta = {
            servicio: "Pendiente" if requerido else "No Aplica"
            for servicio, requerido in ruta_requerida.items()
        }

        self.dias_en_espera_total = 0
        self.fallo_administrativo = False

        es_menor = str(self.edad) in ["0-2", "2-7", "7-12"]
        es_sexual = self.naturaleza == "Violencia Sexual"
        self.prioridad = 1 if (es_menor or es_sexual) else 2

    def step(self):
        if "Pendiente" in self.estado_ruta.values():
            self.dias_en_espera_total += 1
            
            if self.dias_en_espera_total > 30:
                self.fallo_administrativo = True
                
                for servicio in self.estado_ruta:
                    if self.estado_ruta[servicio] == "Pendiente":
                        self.estado_ruta[servicio] = "Caducado (Impunidad)"

    def esta_completado(self):
        return "Pendiente" not in self.estado_ruta.values()


class DependenciaAgente(mesa.Agent):
    def __init__(self, unique_id, model, tipo_servicio, capacidad_diaria):
        super().__init__(unique_id, model)
        self.tipo_servicio = tipo_servicio
        self.capacidad_diaria_base = capacidad_diaria
        self.casos_atendidos_hoy = 0

    def step(self):
        self.casos_atendidos_hoy = 0

    def procesar_cola(self, cola_pendientes, es_fin_semana, pct_fin_semana):
        casos_procesados = 0

        capacidad_hoy = (
            int(self.capacidad_diaria_base * pct_fin_semana)
            if es_fin_semana
            else self.capacidad_diaria_base
        )

        while self.casos_atendidos_hoy < capacidad_hoy and cola_pendientes:
            expediente_actual = cola_pendientes.pop(0)
            expediente_actual.estado_ruta[self.tipo_servicio] = "Atendido"
            self.casos_atendidos_hoy += 1
            casos_procesados += 1

        return casos_procesados
