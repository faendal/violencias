import mesa
import random
from .agents import MujerAgente, InstitucionAgente


class SindemiaViolenciaModel(mesa.Model):
    def __init__(
        self,
        num_mujeres,
        num_inst_salud,
        num_inst_justicia,
        cap_salud,
        cap_justicia,
        df_prophet,
        sim_params,
    ):
        super().__init__()
        self.num_mujeres = num_mujeres
        self.df_prophet = df_prophet.reset_index(drop=True)
        self.sim_params = sim_params
        self.current_step = 0
        self.schedule = mesa.time.RandomActivation(self)

        id_inst = 0
        for _ in range(num_inst_salud):
            self.schedule.add(
                InstitucionAgente(f"Inst_{id_inst}", self, cap_salud, "Salud")
            )
            id_inst += 1
        for _ in range(num_inst_justicia):
            self.schedule.add(
                InstitucionAgente(f"Inst_{id_inst}", self, cap_justicia, "Justicia")
            )
            id_inst += 1

        estratos = list(self.sim_params["dist_estrato"].keys())
        p_estratos = list(self.sim_params["dist_estrato"].values())

        edades = list(self.sim_params["dist_edad"].keys())
        p_edades = list(self.sim_params["dist_edad"].values())

        for i in range(self.num_mujeres):
            estrato = (
                random.choices(estratos, weights=p_estratos)[0]
                if estratos
                else "Sin Dato"
            )
            edad = random.choices(edades, weights=p_edades)[0] if edades else "Sin Dato"
            zona = random.choices(["Urbana", "Rural"], weights=[0.7, 0.3])[
                0
            ]

            self.schedule.add(MujerAgente(i, self, estrato, edad, zona))

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Violencia Real (Ocurrida)": lambda m: sum(
                    [
                        1
                        for a in m.schedule.agents
                        if getattr(a, "violencia_sufrida", False)
                    ]
                ),
                "Capturas SIVIGILA": lambda m: sum(
                    [
                        1
                        for a in m.schedule.agents
                        if getattr(a, "reportado_exitosamente", False)
                    ]
                ),
                "Revictimización": lambda m: sum(
                    [
                        1
                        for a in m.schedule.agents
                        if getattr(a, "veces_violentada", 0) > 1
                    ]
                ),
                "Abandono Geográfico": lambda m: sum(
                    [
                        1
                        for a in m.schedule.agents
                        if getattr(a, "abandono_geografico", False)
                    ]
                ),
                "Abandono Desconfianza": lambda m: sum(
                    [
                        1
                        for a in m.schedule.agents
                        if getattr(a, "abandono_desconfianza", False)
                    ]
                ),
                "Brecha Saturación": lambda m: sum(
                    [
                        1
                        for a in m.schedule.agents
                        if getattr(a, "abandono_saturacion", False)
                        and not getattr(a, "reportado_exitosamente", False)
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

        casos_impunes = sum(
            [
                1
                for a in self.schedule.agents
                if isinstance(a, MujerAgente)
                and a.violencia_sufrida
                and not a.reportado_exitosamente
            ]
        )
        riesgo_diario = (
            (casos_prophet * 1.2) + (casos_impunes * 0.05)
        ) / self.num_mujeres

        nats = list(self.sim_params["dist_naturaleza"].keys())
        p_nats = list(self.sim_params["dist_naturaleza"].values())

        for agent in self.schedule.agents:
            if isinstance(agent, MujerAgente):
                if not agent.violencia_sufrida or agent.reportado_exitosamente:

                    mult = 1.0
                    if str(agent.estrato) in ["1", "2", "3"]:
                        mult *= 1.2
                    if str(agent.edad) in ["12-18"]:
                        mult *= 1.5

                    if random.random() < (riesgo_diario * mult):
                        agent.violencia_sufrida = True
                        agent.veces_violentada += 1
                        agent.reportado_exitosamente = False
                        agent.intentos_fallidos = 0
                        agent.abandono_saturacion = False

                        agent.naturaleza_violencia = (
                            random.choices(nats, weights=p_nats)[0]
                            if nats
                            else "Física"
                        )

        self.schedule.step()
        self.datacollector.collect(self)
        self.current_step += 1
