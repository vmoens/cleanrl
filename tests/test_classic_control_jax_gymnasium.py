import subprocess


def test_dqn_jax():
    subprocess.run(
        "python leanrl/dqn_jax.py --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )


def test_c51_jax():
    subprocess.run(
        "python leanrl/c51_jax.py --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )


def test_c51_jax_eval():
    subprocess.run(
        "python leanrl/c51_jax.py --save-model --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )
