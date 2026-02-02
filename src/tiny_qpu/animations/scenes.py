"""Manim scenes for quantum computing visualization."""
import numpy as np
import sys

try:
    from manim import *
    HAS_MANIM = True
except ImportError:
    HAS_MANIM = False
    print("Manim not installed. Install with: pip install manim")


if HAS_MANIM:

    # =========================================================
    # Scene 1: Bloch Sphere — Qubit Gate Animations
    # =========================================================
    class BlochSphereScene(ThreeDScene):
        """Animate single-qubit gates on the Bloch sphere."""

        def construct(self):
            self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)

            # Title
            title = Text("Qubit on the Bloch Sphere", font_size=36,
                         color=WHITE).to_edge(UP)
            self.add_fixed_in_frame_mobjects(title)

            # Draw sphere
            sphere = Surface(
                lambda u, v: np.array([
                    np.cos(u) * np.cos(v),
                    np.cos(u) * np.sin(v),
                    np.sin(u)
                ]),
                u_range=[-PI / 2, PI / 2],
                v_range=[0, TAU],
                resolution=(24, 48),
                fill_opacity=0.08,
                stroke_color=BLUE_E,
                stroke_width=0.5,
            )
            self.add(sphere)

            # Axes
            axes = ThreeDAxes(
                x_range=[-1.3, 1.3], y_range=[-1.3, 1.3], z_range=[-1.3, 1.3],
                x_length=3, y_length=3, z_length=3,
                axis_config={"color": GREY_B, "stroke_width": 1},
            )
            self.add(axes)

            # Labels
            labels = VGroup(
                Text("|0⟩", font_size=24, color=GREEN).move_to([0, 0, 1.5]),
                Text("|1⟩", font_size=24, color=RED).move_to([0, 0, -1.5]),
                Text("X", font_size=20, color=GREY).move_to([1.5, 0, 0]),
                Text("Y", font_size=20, color=GREY).move_to([0, 1.5, 0]),
            )
            for label in labels:
                self.add_fixed_orientation_mobjects(label)
            self.add(labels)

            # State vector arrow (start at |0⟩)
            state_arrow = Arrow3D(
                start=ORIGIN, end=[0, 0, 1],
                color=YELLOW, thickness=0.03,
            )
            self.play(Create(state_arrow), run_time=1)
            self.wait(0.5)

            # Gate sequence with labels
            gate_sequence = [
                ("H gate", self._h_rotation),
                ("X gate (π)", self._x_rotation),
                ("Z gate", self._z_rotation),
                ("T gate (π/4)", self._t_rotation),
                ("Y gate", self._y_rotation),
            ]

            for gate_name, rotation_func in gate_sequence:
                # Show gate label
                gate_label = Text(gate_name, font_size=30, color=YELLOW)
                gate_label.to_edge(DOWN)
                self.add_fixed_in_frame_mobjects(gate_label)
                self.play(FadeIn(gate_label), run_time=0.3)

                # Animate rotation
                rotation_func(state_arrow)

                self.play(FadeOut(gate_label), run_time=0.3)
                self.wait(0.3)

            self.wait(1)

        def _h_rotation(self, arrow):
            """Hadamard: rotate π around (X+Z)/√2 axis."""
            self.play(
                Rotate(arrow, angle=PI, axis=[1, 0, 1] / np.sqrt(2),
                       about_point=ORIGIN),
                run_time=1.5,
            )

        def _x_rotation(self, arrow):
            """X gate: rotate π around X axis."""
            self.play(
                Rotate(arrow, angle=PI, axis=RIGHT, about_point=ORIGIN),
                run_time=1.5,
            )

        def _z_rotation(self, arrow):
            """Z gate: rotate π around Z axis."""
            self.play(
                Rotate(arrow, angle=PI, axis=UP * 0 + OUT, about_point=ORIGIN),
                run_time=1.5,
            )

        def _t_rotation(self, arrow):
            """T gate: rotate π/4 around Z axis."""
            self.play(
                Rotate(arrow, angle=PI / 4, axis=OUT, about_point=ORIGIN),
                run_time=1.2,
            )

        def _y_rotation(self, arrow):
            """Y gate: rotate π around Y axis."""
            self.play(
                Rotate(arrow, angle=PI, axis=UP, about_point=ORIGIN),
                run_time=1.5,
            )


    # =========================================================
    # Scene 2: Circuit Execution — Step by Step
    # =========================================================
    class CircuitAnimScene(Scene):
        """Animate a quantum circuit being built and executed."""

        def construct(self):
            title = Text("Quantum Circuit — Bell State", font_size=36,
                         color=WHITE).to_edge(UP)
            self.play(Write(title), run_time=0.8)

            # Qubit lines
            q0_line = Line(LEFT * 5, RIGHT * 5, color=BLUE_B).shift(UP * 1)
            q1_line = Line(LEFT * 5, RIGHT * 5, color=BLUE_B).shift(DOWN * 1)
            q0_label = MathTex("|0\\rangle", color=GREEN, font_size=36).next_to(q0_line, LEFT)
            q1_label = MathTex("|0\\rangle", color=GREEN, font_size=36).next_to(q1_line, LEFT)

            self.play(
                Create(q0_line), Create(q1_line),
                Write(q0_label), Write(q1_label),
                run_time=1,
            )

            # State tracker
            state_text = MathTex("|\\psi\\rangle = |00\\rangle",
                                 font_size=30, color=YELLOW).to_edge(DOWN)
            self.play(Write(state_text), run_time=0.5)

            # Gate 1: Hadamard on q0
            h_gate = Square(side_length=0.8, color=PURPLE, fill_opacity=0.3)
            h_gate.move_to(q0_line.get_center() + LEFT * 2)
            h_text = Text("H", font_size=28, color=WHITE).move_to(h_gate)

            self.play(
                FadeIn(h_gate), Write(h_text),
                run_time=0.8,
            )

            # Pulse animation
            pulse = Circle(radius=0.1, color=YELLOW, fill_opacity=0.8)
            pulse.move_to(q0_line.get_left() + RIGHT * 0.5)
            self.play(
                pulse.animate.move_to(h_gate.get_center()),
                run_time=0.6,
            )

            new_state = MathTex(
                "|\\psi\\rangle = \\frac{1}{\\sqrt{2}}(|0\\rangle + |1\\rangle)|0\\rangle",
                font_size=28, color=YELLOW
            ).to_edge(DOWN)
            self.play(
                Transform(state_text, new_state),
                pulse.animate.scale(2).set_opacity(0),
                run_time=0.8,
            )
            self.remove(pulse)

            # Gate 2: CNOT
            cnot_ctrl = Dot(color=BLUE, radius=0.15).move_to(
                q0_line.get_center() + RIGHT * 0.5)
            cnot_targ = Circle(radius=0.3, color=BLUE, stroke_width=3).move_to(
                q1_line.get_center() + RIGHT * 0.5)
            cnot_plus = Cross(stroke_color=BLUE, stroke_width=3).scale(0.15).move_to(
                cnot_targ)
            cnot_line = Line(
                cnot_ctrl.get_center(), cnot_targ.get_center(),
                color=BLUE, stroke_width=2
            )

            self.play(
                FadeIn(cnot_ctrl), Create(cnot_line),
                Create(cnot_targ), Create(cnot_plus),
                run_time=1,
            )

            # Entanglement flash
            flash = VGroup(
                Circle(radius=0.5, color=YELLOW, stroke_width=4).move_to(cnot_ctrl),
                Circle(radius=0.5, color=YELLOW, stroke_width=4).move_to(cnot_targ),
            )
            self.play(
                Create(flash), run_time=0.3,
            )

            bell_state = MathTex(
                "|\\psi\\rangle = \\frac{1}{\\sqrt{2}}(|00\\rangle + |11\\rangle)",
                font_size=28, color=YELLOW
            ).to_edge(DOWN)
            self.play(
                Transform(state_text, bell_state),
                flash.animate.scale(2).set_opacity(0),
                run_time=0.8,
            )
            self.remove(flash)

            # Measurement
            m0_box = Square(side_length=0.8, color=WHITE, fill_opacity=0.2).move_to(
                q0_line.get_center() + RIGHT * 3)
            m1_box = Square(side_length=0.8, color=WHITE, fill_opacity=0.2).move_to(
                q1_line.get_center() + RIGHT * 3)
            m0_text = Text("M", font_size=24, color=WHITE).move_to(m0_box)
            m1_text = Text("M", font_size=24, color=WHITE).move_to(m1_box)

            self.play(
                FadeIn(m0_box), FadeIn(m1_box),
                Write(m0_text), Write(m1_text),
                run_time=0.8,
            )

            # Collapse animation
            result = MathTex(
                "|00\\rangle \\text{ or } |11\\rangle \\text{ with 50/50 probability}",
                font_size=28, color=GREEN
            ).to_edge(DOWN)

            self.play(
                Transform(state_text, result),
                m0_box.animate.set_fill(GREEN, opacity=0.5),
                m1_box.animate.set_fill(GREEN, opacity=0.5),
                run_time=1,
            )

            self.wait(2)


    # =========================================================
    # Scene 3: VQE Energy Optimization
    # =========================================================
    class VQEOptimScene(Scene):
        """Animate VQE optimization on an energy landscape."""

        def construct(self):
            title = Text("VQE Optimization — H₂ Ground State", font_size=34,
                         color=WHITE).to_edge(UP)
            self.play(Write(title), run_time=0.8)

            # Energy landscape
            axes = Axes(
                x_range=[-3, 3, 1], y_range=[-1.2, 0.5, 0.5],
                x_length=10, y_length=5,
                axis_config={"color": GREY_B},
            ).shift(DOWN * 0.3)

            x_label = axes.get_x_axis_label("\\theta")
            y_label = axes.get_y_axis_label("E(\\theta)")

            # Energy function with local minima
            def energy_func(x):
                return -0.8 * np.exp(-x**2) - 0.3 * np.exp(-(x-2)**2) + 0.1 * np.sin(3*x)

            landscape = axes.plot(energy_func, x_range=[-3, 3], color=BLUE_C,
                                  stroke_width=3)

            self.play(Create(axes), Write(x_label), Write(y_label), run_time=1)
            self.play(Create(landscape), run_time=1)

            # Exact ground state line
            exact_line = axes.plot(lambda x: -0.8, x_range=[-3, 3],
                                   color=GREEN, stroke_width=1.5)
            exact_label = Text("Exact: -0.80 Ha", font_size=20,
                              color=GREEN).next_to(exact_line, RIGHT, buff=0.3).shift(UP*0.3)
            self.play(Create(exact_line), Write(exact_label), run_time=0.5)

            # Chemical accuracy band
            chem_band = axes.get_area(
                axes.plot(lambda x: -0.8 + 0.05, x_range=[-3, 3]),
                bounded_graph=axes.plot(lambda x: -0.8 - 0.05, x_range=[-3, 3]),
                color=GREEN, opacity=0.15,
            )
            chem_label = Text("Chemical\naccuracy", font_size=16,
                             color=GREEN_A).move_to(axes.c2p(-2.5, -0.8))
            self.play(FadeIn(chem_band), Write(chem_label), run_time=0.5)

            # Optimizer dot
            start_x = 2.5
            dot = Dot(axes.c2p(start_x, energy_func(start_x)),
                     color=YELLOW, radius=0.12)
            dot_label = Text("VQE", font_size=20, color=YELLOW).next_to(dot, UP)

            self.play(FadeIn(dot), Write(dot_label), run_time=0.5)

            # Optimization trajectory
            trajectory_x = [2.5, 2.2, 1.8, 1.3, 0.8, 0.4, 0.15, 0.05, 0.01, 0.0]
            for x in trajectory_x[1:]:
                new_pos = axes.c2p(x, energy_func(x))
                self.play(
                    dot.animate.move_to(new_pos),
                    dot_label.animate.next_to(Dot().move_to(new_pos), UP),
                    run_time=0.4,
                )

            # Success flash
            success = Text("✓ Chemical Accuracy Achieved!", font_size=28,
                          color=GREEN).to_edge(DOWN)
            self.play(
                Write(success),
                Flash(dot, color=GREEN, num_lines=12),
                run_time=1,
            )

            self.wait(2)


    # =========================================================
    # Scene 4: Potential Energy Surface Animation
    # =========================================================
    class PESAnimScene(Scene):
        """Animate H2 potential energy surface with VQE points."""

        def construct(self):
            title = Text("H₂ Potential Energy Surface", font_size=34,
                         color=WHITE).to_edge(UP)
            self.play(Write(title), run_time=0.8)

            axes = Axes(
                x_range=[0.2, 3.0, 0.5], y_range=[-1.2, -0.3, 0.2],
                x_length=10, y_length=5.5,
                axis_config={"color": GREY_B},
            ).shift(DOWN * 0.3)

            x_label = axes.get_x_axis_label("R (\\AA)")
            y_label = axes.get_y_axis_label("E (Ha)")

            self.play(Create(axes), Write(x_label), Write(y_label), run_time=1)

            # Pre-computed H2 energies
            bond_lengths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.735, 0.8, 0.9,
                           1.0, 1.1, 1.2, 1.5, 2.0, 2.5, 3.0]
            energies = [-0.383, -0.648, -0.823, -0.942, -1.023, -1.078,
                       -1.094, -1.105, -1.116, -1.117, -1.110, -1.098,
                       -1.055, -0.982, -0.936, -0.910]

            # Animate curve building point by point
            dots = VGroup()
            for R, E in zip(bond_lengths, energies):
                dot = Dot(axes.c2p(R, E), color=BLUE, radius=0.06)
                self.play(FadeIn(dot), run_time=0.15)
                dots.add(dot)

            # Connect with smooth curve
            points = [axes.c2p(R, E) for R, E in zip(bond_lengths, energies)]
            curve = VMobject(color=BLUE_C, stroke_width=3)
            curve.set_points_smoothly(points)
            self.play(Create(curve), run_time=1)

            # Highlight minimum
            min_idx = energies.index(min(energies))
            min_dot = Dot(axes.c2p(bond_lengths[min_idx], energies[min_idx]),
                         color=YELLOW, radius=0.15)
            min_label = MathTex(
                f"R_{{eq}} = {bond_lengths[min_idx]:.1f}\\,\\AA",
                font_size=24, color=YELLOW
            ).next_to(min_dot, DOWN + RIGHT)

            self.play(
                FadeIn(min_dot),
                Write(min_label),
                Flash(min_dot, color=YELLOW, num_lines=8),
                run_time=1,
            )

            # Add dissociation label
            dissoc = Text("Dissociation →", font_size=20,
                         color=RED_C).move_to(axes.c2p(2.5, -0.85))
            self.play(Write(dissoc), run_time=0.5)

            # Bond formation label
            repuls = Text("← Repulsion", font_size=20,
                         color=RED_C).move_to(axes.c2p(0.35, -0.55))
            self.play(Write(repuls), run_time=0.5)

            self.wait(2)


    # =========================================================
    # Scene 5: Noise Degradation
    # =========================================================
    class NoiseDegradScene(Scene):
        """Visualize how noise degrades quantum computation."""

        def construct(self):
            title = Text("Noise Destroys Quantum Advantage", font_size=34,
                         color=WHITE).to_edge(UP)
            self.play(Write(title), run_time=0.8)

            # Two density matrix grids side by side
            clean_title = Text("Clean", font_size=24, color=GREEN).shift(LEFT * 3 + UP * 2)
            noisy_title = Text("Noisy (p=0.10)", font_size=24, color=RED).shift(RIGHT * 3 + UP * 2)
            self.play(Write(clean_title), Write(noisy_title), run_time=0.5)

            # Clean Bell state density matrix
            clean_dm = np.array([
                [0.5, 0, 0, 0.5],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0.5, 0, 0, 0.5],
            ])

            # Noisy version
            noisy_dm = np.array([
                [0.47, 0, 0, 0.41],
                [0, 0.03, 0, 0],
                [0, 0, 0.03, 0],
                [0.41, 0, 0, 0.47],
            ])

            clean_grid = self._make_dm_grid(clean_dm, GREEN).shift(LEFT * 3)
            noisy_grid = self._make_dm_grid(noisy_dm, RED).shift(RIGHT * 3)

            self.play(Create(clean_grid), run_time=1)
            self.play(Create(noisy_grid), run_time=1)

            # Metrics
            metrics = VGroup(
                Text(f"Fidelity:  1.000 → 0.910", font_size=22, color=YELLOW),
                Text(f"Purity:    1.000 → 0.856", font_size=22, color=YELLOW),
                Text(f"Concurrence: 1.0 → 0.82", font_size=22, color=YELLOW),
            ).arrange(DOWN, aligned_edge=LEFT).to_edge(DOWN)

            for m in metrics:
                self.play(Write(m), run_time=0.5)

            self.wait(2)

        def _make_dm_grid(self, dm, color):
            grid = VGroup()
            for i in range(4):
                for j in range(4):
                    val = dm[i, j]
                    opacity = abs(val)
                    cell = Square(
                        side_length=0.55,
                        fill_color=color,
                        fill_opacity=opacity,
                        stroke_color=WHITE,
                        stroke_width=0.5,
                    ).shift(RIGHT * j * 0.6 + DOWN * i * 0.6)

                    if abs(val) > 0.01:
                        label = Text(f"{val:.2f}", font_size=14, color=WHITE)
                        label.move_to(cell)
                        grid.add(VGroup(cell, label))
                    else:
                        grid.add(cell)

            grid.center()
            return grid


def render_scene(scene_name: str, quality: str = 'low',
                 output_dir: str = '.'):
    """Render a Manim scene programmatically."""
    import subprocess
    scene_file = __file__.replace('__init__', 'scenes')
    quality_flag = {'low': '-pql', 'medium': '-pqm', 'high': '-pqh'}
    flag = quality_flag.get(quality, '-pql')
    cmd = f'manim {flag} {scene_file} {scene_name}'
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)
