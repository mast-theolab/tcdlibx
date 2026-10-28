"""
Animated particles moving along streamlines.
"""
import math
from typing import TypedDict

from tcdlibx.utils.vtk_utils import vtk


class ParticleData(TypedDict):
    """State of one animated particle."""
    actor: vtk.vtkActor
    points: list[tuple[float, float, float]]
    position: float  # normalized position along the streamline (0-1)
    speed: float
    active: bool
    transform: vtk.vtkTransform | None


class StreamlineParticleAnimator:
    """
    Handles animated particles along streamlines in VTK.
    """

    def __init__(self, renderer: vtk.vtkRenderer, particle_type: str = 'sphere') -> None:
        """
        Initialize the particle animator.

        Args:
            renderer: VTK renderer to add particles to
            particle_type: Type of particle to use ("sphere" or "cone")
        """
        self.renderer = renderer
        self.particle_actors: list[ParticleData] = []
        self.is_animating = False
        self.particle_type = particle_type

    def create_particles(self, streamline_polydata: vtk.vtkPolyData,
                         num_particles: int = 5) -> list[ParticleData]:
        """
        Create animated particles along streamlines.

        Args:
            streamline_polydata: VTK polydata containing streamlines
            num_particles: Maximum number of particles to create

        Returns:
            List of particle data dictionaries
        """
        particles: list[ParticleData] = []

        # Extract streamlines from polydata
        streamlines = self._extract_streamlines(streamline_polydata)

        # Create particles for each streamline
        for i, line_points in enumerate(streamlines):
            if i >= num_particles:  # Limit number of particles
                break

            if len(line_points) < 2:
                continue

            # Create particle actor based on type
            particle_actor = self._create_particle_actor()

            # Store particle data
            particle_data: ParticleData = {
                'actor': particle_actor,
                'points': line_points,
                'position': 0.0,
                'speed': 0.003 + i * 0.001,  # Slower speeds for better visibility
                'active': True,
                'transform': vtk.vtkTransform() if self.particle_type == "cone" else None
            }

            # Set initial position and orientation
            if len(line_points) > 0:
                # For cones, set orientation first, then position
                if self.particle_type == "cone" and len(line_points) > 1:
                    direction = self._calculate_direction(line_points[0], line_points[1])
                    self._orient_cone(particle_actor, direction, particle_data['transform'])

                # Set position after orientation for cones
                particle_actor.SetPosition(line_points[0])

            particles.append(particle_data)
            self.renderer.AddActor(particle_actor)

        self.particle_actors = particles
        return particles

    def _extract_streamlines(self, streamline_polydata: vtk.vtkPolyData) -> list[list[tuple[float, float, float]]]:
        """
        Extract individual streamlines from VTK polydata.

        Args:
            streamline_polydata: VTK polydata containing streamlines

        Returns:
            List of streamlines, each as a list of 3D points
        """
        streamlines = []
        streamline_polydata.GetLines().InitTraversal()

        id_list = vtk.vtkIdList()
        while streamline_polydata.GetLines().GetNextCell(id_list):
            line_points = []
            for i in range(id_list.GetNumberOfIds()):
                point_id = id_list.GetId(i)
                point = streamline_polydata.GetPoint(point_id)
                line_points.append(point)

            if len(line_points) > 1:
                streamlines.append(line_points)

        return streamlines

    def _create_particle_actor(self) -> vtk.vtkActor:
        """
        Create a particle actor based on the particle type.

        Returns:
            VTK actor for the particle
        """
        if self.particle_type == "cone":
            return self._create_cone_particle()
        else:
            return self._create_sphere_particle()

    def _create_sphere_particle(self) -> vtk.vtkActor:
        """
        Create a glowing sphere actor for a particle.

        Returns:
            VTK actor for the particle
        """
        # Create sphere
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(0.03)  # Radius appropriate for atomic units scale
        sphere.SetPhiResolution(16)
        sphere.SetThetaResolution(16)

        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        # Create actor with glowing appearance
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0.5, 0)  # Bright orange
        actor.GetProperty().SetSpecular(0.9)
        actor.GetProperty().SetSpecularPower(100)
        actor.GetProperty().SetAmbient(0.7)  # Higher ambient light for glow effect
        actor.GetProperty().SetDiffuse(0.9)  # Higher diffuse for brightness

        return actor

    def _create_cone_particle(self) -> vtk.vtkActor:
        """
        Create a cone actor for a particle.

        Returns:
            VTK actor for the particle
        """
        # Create cone
        cone = vtk.vtkConeSource()
        cone.SetRadius(0.06)  # Increased radius for better visibility
        cone.SetHeight(0.12)  # Increased height for better visibility
        cone.SetResolution(12)
        cone.SetDirection(1, 0, 0)  # Default direction along X-axis
        cone.SetCenter(0, 0, 0)  # Center the cone at origin

        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cone.GetOutputPort())

        # Create actor with glowing appearance
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0, 0.8, 1)  # Bright cyan for cones
        actor.GetProperty().SetSpecular(0.9)
        actor.GetProperty().SetSpecularPower(100)
        actor.GetProperty().SetAmbient(0.7)  # Higher ambient light for glow effect
        actor.GetProperty().SetDiffuse(0.9)  # Higher diffuse for brightness

        # Store reference to cone source for later direction updates
        actor.cone_source = cone  # pyright: ignore[reportAttributeAccessIssue]

        return actor

    def _calculate_direction(self, p1: tuple, p2: tuple) -> tuple:
        """
        Calculate the direction vector from p1 to p2.

        Args:
            p1: First point (x, y, z)
            p2: Second point (x, y, z)

        Returns:
            Normalized direction vector
        """
        # Calculate direction vector
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]

        # Normalize
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length > 0:
            return (dx/length, dy/length, dz/length)
        else:
            return (1, 0, 0)  # Default direction

    def _orient_cone(self, actor: vtk.vtkActor, direction: tuple, transform: vtk.vtkTransform) -> None:
        """
        Orient the cone to point in the given direction.

        Args:
            actor: VTK actor to orient
            direction: Direction vector (x, y, z)
            transform: VTK transform object to store the transformation
        """
        # Normalize direction vector
        dx, dy, dz = direction
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length > 0:
            dx, dy, dz = dx/length, dy/length, dz/length
        else:
            dx, dy, dz = 1, 0, 0  # Default direction

        # Get the cone source from the stored reference
        if hasattr(actor, 'cone_source'):
            cone_source = actor.cone_source
            # Set the cone direction directly in the source
            cone_source.SetDirection(dx, dy, dz)
            cone_source.Update()

        # Don't apply any transform to the actor - let it use world coordinates directly
        actor.SetUserTransform(None)

    def update_particles(self) -> None:
        """
        Update particle positions along their streamlines.
        """
        if not self.particle_actors:
            return

        for particle in self.particle_actors:
            if not particle['active']:
                continue

            points = particle['points']
            if len(points) < 2:
                continue

            # Update position
            particle['position'] += particle['speed']
            if particle['position'] >= 1.0:
                particle['position'] = 0.0

            # Interpolate position along streamline
            interpolated_pos = self._interpolate_position(points, particle['position'])

            # For cones, update orientation first, then position
            if self.particle_type == "cone" and particle['transform'] is not None:
                direction = self._get_direction_at_position(points, particle['position'])
                self._orient_cone(particle['actor'], direction, particle['transform'])

            # Set position after orientation for cones
            particle['actor'].SetPosition(interpolated_pos)

    def _get_direction_at_position(self, points: list, position: float) -> tuple:
        """
        Get the direction vector at a specific position along the streamline.

        Args:
            points: List of 3D points defining the streamline
            position: Normalized position along streamline (0-1)

        Returns:
            Direction vector at the given position
        """
        total_segments = len(points) - 1
        segment_pos = position * total_segments
        segment_idx = int(segment_pos)

        if segment_idx >= total_segments:
            segment_idx = total_segments - 1

        # Get direction from current segment
        if segment_idx + 1 < len(points):
            p1 = points[segment_idx]
            p2 = points[segment_idx + 1]
            return self._calculate_direction(p1, p2)
        else:
            # Use previous segment direction
            if segment_idx > 0:
                p1 = points[segment_idx - 1]
                p2 = points[segment_idx]
                return self._calculate_direction(p1, p2)
            else:
                return (1, 0, 0)  # Default direction

    def _interpolate_position(self, points: list, position: float) -> list:
        """
        Interpolate position along a streamline.

        Args:
            points: List of 3D points defining the streamline
            position: Normalized position along streamline (0-1)

        Returns:
            Interpolated 3D position
        """
        total_segments = len(points) - 1
        segment_pos = position * total_segments
        segment_idx = int(segment_pos)
        local_pos = segment_pos - segment_idx

        if segment_idx >= total_segments:
            segment_idx = total_segments - 1
            local_pos = 1.0

        # Linear interpolation between points
        p1 = points[segment_idx]
        p2 = points[segment_idx + 1] if segment_idx + 1 < len(points) else points[segment_idx]

        return [
            p1[0] + local_pos * (p2[0] - p1[0]),
            p1[1] + local_pos * (p2[1] - p1[1]),
            p1[2] + local_pos * (p2[2] - p1[2])
        ]

    def start_animation(self) -> None:
        """
        Start the particle animation.
        """
        self.is_animating = True
        for particle in self.particle_actors:
            particle['active'] = True

    def stop_animation(self) -> None:
        """
        Stop the particle animation and remove all particles.
        """
        self.is_animating = False

        # Remove all particle actors from renderer
        for particle in self.particle_actors:
            self.renderer.RemoveActor(particle['actor'])
            particle['active'] = False

        self.particle_actors = []

    def set_particle_speed(self, speed_multiplier: float) -> None:
        """
        Adjust the speed of all particles.

        Args:
            speed_multiplier: Multiplier for particle speeds
        """
        for i, particle in enumerate(self.particle_actors):
            base_speed = 0.003 + i * 0.001
            particle['speed'] = base_speed * speed_multiplier

    def set_particle_visibility(self, visible: bool) -> None:
        """
        Show or hide all particles.

        Args:
            visible: Whether particles should be visible
        """
        for particle in self.particle_actors:
            particle['actor'].SetVisibility(visible)

    def update_particle_type(self, new_particle_type: str, streamline_polydata: vtk.vtkPolyData, num_particles: int | None = None) -> None:
        """
        Update the particle type for existing animation.

        Args:
            new_particle_type: New particle type ("sphere" or "cone")
            streamline_polydata: VTK polydata containing streamlines
            num_particles: New number of particles (optional, keeps current if None)
        """
        if new_particle_type == self.particle_type and (num_particles is None or num_particles == len(self.particle_actors)):
            return  # No change needed

        # Store animation state
        was_animating = self.is_animating

        # Keep the current count: stop_animation() empties particle_actors
        if num_particles is None:
            num_particles = len(self.particle_actors) or 5

        # Stop current animation and remove particles
        self.stop_animation()

        # Update particle type
        self.particle_type = new_particle_type

        # Create new particles with new type
        self.create_particles(streamline_polydata, num_particles)

        # Restart animation if it was running
        if was_animating:
            self.start_animation()

    def get_particle_count(self) -> int:
        """
        Get the current number of particles.

        Returns:
            Number of active particles
        """
        return len(self.particle_actors)

    def force_render_update(self) -> None:
        """
        Force the renderer to update the display.
        """
        if self.renderer.GetRenderWindow() is not None:
            self.renderer.GetRenderWindow().Render()


def create_streamline_particles(renderer: vtk.vtkRenderer,
                                polydata: vtk.vtkPolyData,
                                num_particles: int = 10,
                                particle_type: str = "sphere") -> StreamlineParticleAnimator:
    """
    Factory function to create a StreamlineParticleAnimator instance.

    Args:
        renderer: VTK renderer to add particles to
        polydata: VTK polydata containing streamline data
        num_particles: Number of particles to create
        particle_type: Type of particles ("sphere" or "cone")

    Returns:
        StreamlineParticleAnimator instance with particles created
    """
    animator = StreamlineParticleAnimator(renderer, particle_type)

    # Only create particles if polydata has lines/points
    if polydata.GetNumberOfPoints() > 0 and polydata.GetNumberOfCells() > 0:
        animator.create_particles(polydata, num_particles)

    return animator
