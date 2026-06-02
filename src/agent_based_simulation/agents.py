import mesa
import random


class MujerAgente(mesa.Agent):
    def __init__(self, unique_id, model, estrato, edad, zona):
        super().__init__(unique_id, model)
        self.estrato = estrato
        self.edad = edad
        self.zona = zona

        self.violencia_sufrida = False
        self.naturaleza_violencia = None
        self.reportado_exitosamente = False
        self.veces_violentada = 0

        self.intentos_fallidos = 0
        self.abandono_geografico = False
        self.abandono_desconfianza = False
        self.abandono_saturacion = False

    def step(self):
        if self.violencia_sufrida and not self.reportado_exitosamente:
            if self.abandono_desconfianza or self.abandono_geografico:
                return

            if self.zona == "Rural" and random.random() < 0.65:
                self.abandono_geografico = True
                return

            tipo_institucion_buscada = (
                "Salud" if self.naturaleza_violencia == "Sexual" else "Justicia"
            )

            instituciones = [
                ag
                for ag in self.model.schedule.agents
                if isinstance(ag, InstitucionAgente)
                and ag.tipo == tipo_institucion_buscada
            ]

            if instituciones:
                institucion = random.choice(instituciones)
                resultado = institucion.recibir_denuncia()

                if resultado == "Atendido":
                    self.reportado_exitosamente = True
                    self.abandono_saturacion = False
                elif resultado == "Rechazado":
                    self.intentos_fallidos += 1
                    self.abandono_saturacion = True

                    if self.intentos_fallidos >= 3:
                        self.abandono_desconfianza = True


class InstitucionAgente(mesa.Agent):
    def __init__(self, unique_id, model, capacidad_diaria, tipo):
        super().__init__(unique_id, model)
        self.capacidad_diaria = capacidad_diaria
        self.tipo = tipo
        self.casos_atendidos_hoy = 0

    def step(self):
        self.casos_atendidos_hoy = 0

    def recibir_denuncia(self):
        if self.casos_atendidos_hoy < self.capacidad_diaria:
            self.casos_atendidos_hoy += 1
            return "Atendido"
        return "Rechazado"
