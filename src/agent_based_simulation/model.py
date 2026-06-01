import mesa
import random
from .agents import MujerAgente, InstitucionAgente


class SindemiaViolenciaModel(mesa.Model):
    def __init__(
        self, num_mujeres, num_instituciones, capacidad_institucional, df_prophet
    ):
        super().__init__()
        self.num_mujeres = num_mujeres
        self.df_prophet = df_prophet.reset_index(drop=True)
        self.current_step = 0
        self.schedule = mesa.time.RandomActivation(self)

        for i in range(num_instituciones):
            inst = InstitucionAgente(f"Inst_{i}", self, capacidad_institucional)
            self.schedule.add(inst)

        for i in range(self.num_mujeres):
            estrato = random.choices(
                [1, 2, 3, 4, 5, 6], weights=[0.1, 0.4, 0.3, 0.1, 0.05, 0.05]
            )[0]
            edad = random.randint(5, 80)
            zona = random.choices(["Urbana", "Rural"], weights=[0.7, 0.3])[0]
            mujer = MujerAgente(i, self, estrato, edad, zona)
            self.schedule.add(mujer)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Violencia Real (Ocurrida)": lambda m: sum(
                    [
                        1
                        for a in m.schedule.agents
                        if getattr(a, "violencia_sufrida", False)
                    ]
                ),
                "Casos Capturados (SIVIGILA)": lambda m: sum(
                    [
                        1
                        for a in m.schedule.agents
                        if getattr(a, "reportado_exitosamente", False)
                    ]
                ),
            }
        )

    def step(self):
        if self.current_step < len(self.df_prophet):
            casos_prophet = self.df_prophet.loc[
                self.current_step, "Casos Pronosticados"
            ]
        else:
            casos_prophet = 0

        riesgo_diario = (casos_prophet * 1.5) / self.num_mujeres

        for agent in self.schedule.agents:
            if isinstance(agent, MujerAgente):
                agent.violencia_sufrida = False

                mult = 1.0
                if agent.estrato in [2, 3]:
                    mult *= 1.2
                if 10 <= agent.edad <= 14:
                    mult *= 1.5

                if random.random() < (riesgo_diario * mult):
                    agent.violencia_sufrida = True
                    agent.reportado_exitosamente = False

        self.schedule.step()
        self.datacollector.collect(self)
        self.current_step += 1
