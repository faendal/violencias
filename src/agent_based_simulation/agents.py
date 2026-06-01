import mesa
import random


class MujerAgente(mesa.Agent):
    def __init__(self, unique_id, model, estrato, edad, zona):
        super().__init__(unique_id, model)
        self.estrato = estrato
        self.edad = edad
        self.zona = zona
        self.violencia_sufrida = False
        self.reportado_exitosamente = False

    def step(self):
        if self.violencia_sufrida and not self.reportado_exitosamente:
            if self.zona == "Rural" and random.random() < 0.65:
                return

            instituciones = [
                ag
                for ag in self.model.schedule.agents
                if isinstance(ag, InstitucionAgente)
            ]
            if instituciones:
                institucion = random.choice(instituciones)
                if institucion.recibir_denuncia():
                    self.reportado_exitosamente = True


class InstitucionAgente(mesa.Agent):
    def __init__(self, unique_id, model, capacidad_diaria):
        super().__init__(unique_id, model)
        self.capacidad_diaria = capacidad_diaria
        self.casos_atendidos_hoy = 0

    def step(self):
        self.casos_atendidos_hoy = 0

    def recibir_denuncia(self):
        if self.casos_atendidos_hoy < self.capacidad_diaria:
            self.casos_atendidos_hoy += 1
            return True
        return False
