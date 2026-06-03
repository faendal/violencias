import mesa


class ExpedienteAgente(mesa.Agent):
    def __init__(self, unique_id, model, estrato, edad, naturaleza, ruta_requerida):
        super().__init__(unique_id, model)
        # ADN Empírico del caso
        self.estrato = estrato
        self.edad = edad
        self.naturaleza = naturaleza

        # Diccionario de servicios que el caso NECESITA (True/False) según SIVIGILA
        self.ruta_requerida = ruta_requerida

        # Diccionario de estado actual ('Pendiente', 'Atendido', 'No Aplica')
        self.estado_ruta = {
            servicio: "Pendiente" if requerido else "No Aplica"
            for servicio, requerido in ruta_requerida.items()
        }

        # Memoria individual del expediente (Teoría de colas)
        self.dias_en_espera_total = 0
        self.fallo_administrativo = False

    def step(self):
        # Si tiene algún servicio pendiente, el expediente envejece en el sistema
        if "Pendiente" in self.estado_ruta.values():
            self.dias_en_espera_total += 1

            # Si un expediente pasa más de 30 días sin completarse, se considera impunidad/fallo
            if self.dias_en_espera_total > 30:
                self.fallo_administrativo = True

    def esta_completado(self):
        return "Pendiente" not in self.estado_ruta.values()


class DependenciaAgente(mesa.Agent):
    def __init__(self, unique_id, model, tipo_servicio, capacidad_diaria):
        super().__init__(unique_id, model)
        self.tipo_servicio = tipo_servicio  # Ej: 'Salud Mental', 'Proteccion'
        self.capacidad_diaria = capacidad_diaria
        self.casos_atendidos_hoy = 0

    def step(self):
        # Al inicio de cada día, la dependencia renueva sus cupos
        self.casos_atendidos_hoy = 0

    def procesar_cola(self, cola_pendientes):
        """Toma la lista de expedientes que requieren este servicio y los procesa (FIFO)"""
        casos_procesados = 0

        while self.casos_atendidos_hoy < self.capacidad_diaria and cola_pendientes:
            # Extraer el caso más antiguo de la cola
            expediente_actual = cola_pendientes.pop(0)

            # Marcar este servicio como atendido en el expediente
            expediente_actual.estado_ruta[self.tipo_servicio] = "Atendido"

            self.casos_atendidos_hoy += 1
            casos_procesados += 1

        return casos_procesados
