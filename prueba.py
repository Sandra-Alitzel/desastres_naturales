# run_training.py

import subprocess
import sys
from pathlib import Path

def main():
    """
    Ejecuta el script de entrenamiento del modelo (train_model.py).
    """
    # Define la ruta al script de entrenamiento
    ROOT_DIR = Path(__file__).resolve().parent
    train_script_path = ROOT_DIR / "scripts" / "train_model.py"

    if not train_script_path.exists():
        print(f"Error: No se encontró el script de entrenamiento en {train_script_path}")
        sys.exit(1)

    print("=====================================================")
    print(f"  Iniciando Entrenamiento del Modelo: {train_script_path.name}")
    print("  Esto ejecutará 'build_dataset' con el nuevo balanceo.")
    print("=====================================================")

    # Comando para ejecutar el script usando el intérprete actual de Python
    command = [
        sys.executable,  # Usa el mismo intérprete de Python que ejecuta este script
        str(train_script_path)
    ]

    try:
        # Ejecuta el script y captura la salida en tiempo real
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Imprime la salida del subproceso en tiempo real
        while process.poll() is None:
            output = process.stdout.readline()
            if output:
                print(output.strip())

        # Asegura que cualquier salida restante se imprima
        for output in process.stdout.readlines():
            print(output.strip())

        return_code = process.wait()

        if return_code == 0:
            print("\n=====================================================")
            print("  ✅ Entrenamiento del modelo completado exitosamente.")
            print("  Revisa 'models/damage_metrics.json' para el performance.")
            print("=====================================================")
        else:
            print(f"\n=====================================================")
            print(f"  ❌ Error: El script terminó con código {return_code}")
            print("=====================================================")

    except Exception as e:
        print(f"\n❌ Ocurrió un error al intentar ejecutar el script: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()