from envs.particle_envs import PointParticlePosition, PointParticleConstantVelocity, PointParticleRandomWalkPosition, PointParticleRandomWalkVelocity, PointParticleRandomWalkAccel, PointParticleLissajousTracking
from envs.astrobee_envs import SE3QuadFullyActuatedPosition, SE3QuadFullyActuatedPosition, SE3QuadFullyActuatedRandomWalk, SE3QuadFullyActuatedLissajous
from envs.quadrotor_envs import SE2xRQuadPosition, SE2xRQuadRandomWalk, SE2xRQuadLissajous

def get_env_by_name(name, **kwargs):
    """ Return an environment by name, and passes kwargs to the constructor """

    # Particle Envs:
    if "particle" in name:
        if "position" in name:
            env = PointParticlePosition(**kwargs)
        elif "constant_velocity" in name:
            env = PointParticleConstantVelocity(**kwargs)
        elif "random_walk_position" in name:
            env = PointParticleRandomWalkPosition(**kwargs)
        elif "random_walk_velocity" in name:
            env = PointParticleRandomWalkVelocity(**kwargs)
        elif "random_walk_accel" in name:
            env = PointParticleRandomWalkAccel(**kwargs)
        elif "random_lissajous" in name:
            env = PointParticleLissajousTracking(**kwargs)
        else:
            raise ValueError("Invalid environment name: {name}")
    
    # Astrobee Envs:
    elif "astrobee" in name:
        if "position" in name:
            env = SE3QuadFullyActuatedPosition(**kwargs)
        elif "random_walk" in name:
            env = SE3QuadFullyActuatedRandomWalk(**kwargs)
        elif "lissajous" in name:
            env = SE3QuadFullyActuatedLissajous(**kwargs)
        else:
            raise ValueError("Invalid environment name: {name}")
    
    # Quadrotor Envs:
    elif "quadrotor" in name:
        if "position" in name:
            env = SE2xRQuadPosition(**kwargs)
        elif "random_walk" in name:
            env = SE2xRQuadRandomWalk(**kwargs)
        elif "lissajous" in name:
            env = SE2xRQuadLissajous(**kwargs)
        else:
            raise ValueError("Invalid environment name: {name}")
    
    else:
        raise ValueError(f"Unknown environment name: {name}")
    
    return env