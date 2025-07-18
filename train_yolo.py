import os
import yaml
import wandb
from ultralytics import YOLO
from codecarbon import EmissionsTracker
import logging
import sys
import time
from pathlib import Path
import argparse
import traceback

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Directory di base
BASE_DIR = Path('/app')

# Directory per i risultati e i checkpoint
RESULTS_DIR = BASE_DIR / 'results'

# Assicurati che le directory esistano
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_yaml(file_path):
    """Carica un singolo file YAML."""
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    logging.info(f"Configurazione caricata: {file_path}")
    return data


def save_co2_data(co2_data, run_results_dir):
    """Salva i dati di produzione di CO2 in un file locale."""
    co2_file = run_results_dir / 'co2_production.txt'
    with open(co2_file, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - CO2: {co2_data} kg\n")
    logging.info(f"CO2 salvato in {co2_file}")


def test_model(model, run_name):
    """Esegue il test del modello sul dataset di test."""
    logging.info("Inizio del test del modello...")
    try:
        # Esegui il test utilizzando il set di test definito nel dataset YAML
        results = model.val(
            split="test",
            save_txt=True,
            save_json=True,
            save_conf=True,
            name=f"{run_name}_test",
            verbose=True
        )

    except Exception as e:
        logging.error(f"Errore durante il test del modello: {e}")
        raise e


def train_model(model_config_path, dataset_config_path, tracker=None):
    """Esegue l'addestramento del modello YOLOv10 con le configurazioni specificate."""
    #run_name = "undefined" 
    try:
        model_config = load_yaml(model_config_path)
        dataset_config = load_yaml(dataset_config_path)

        # Configura i nomi per il salvataggio dei risultati
        config_name = Path(model_config_path).stem
        dataset_name = Path(dataset_config_path).stem
        run_name = f"{config_name}_{dataset_name}"
        run_results_dir = RESULTS_DIR / run_name
        os.makedirs(run_results_dir, exist_ok=True)

        try:
            # Inizializza il tracker di CO2
            tracker = EmissionsTracker(project_name=f"YOLOv10_{run_name}", output_dir=str(run_results_dir))
            tracker.start()
        except Exception as e:
            logging.error(f" 1 Errore durante l'addestramento di {run_name}: {e}")

        def check_for_test_tag(name: str):
            return "Test" in name or "test" in name

        # combined_config = {**model_config, **dataset_config}

        # Verifica e imposta i tag
        if check_for_test_tag(run_name):
            wandb.init(
                project="results",
                name=f"{run_name}",
                tags=["test"]
            )
        else:
            wandb.init(
                project="results",
                name=f"{run_name}",
            )


        # wandb.define_metric("CO2_production_kg", summary="max")

        # Carica il modello con Ultralytics
        model_architecture_path = str(model_config_path)
        model = YOLO(model_architecture_path)
        # add_wandb_callback(model, enable_model_checkpointing=True)

        try:
            model.train(
                data=str(dataset_config_path),
                name=f"{run_name}",
                resume=True,
                **model_config.get('training', {})
            )
        except Exception as e:
            logging.error(f" 3 Errore durante l'addestramento di {run_name} errore: {e}")


        try:
            # Termina il tracker di CO2 e ottieni i dati
            emissions = tracker.stop()
            # wandb.run.summary["CO2_production_kg"] = emissions
            # wandb.log({"CO2_production_kg": emissions})
            logging.info(f"Emissioni di CO2 per {run_name}: {emissions} kg")

            # Registra i dati di CO2 su wandb


            # Salva i dati di CO2 in un file locale
            save_co2_data(emissions, run_results_dir)
        except Exception as e:
            logging.error(f" 5 Errore durante l'addestramento di {run_name}: {e}")
            raise e


        try:
            # Test del modello
            test_model(model, run_name)

        except Exception as e:
            logging.error(f" 6 Errore durante l'addestramento di {run_name}: {e}")
            wandb.finish()
            raise e

        finally:
            try:
                wandb.finish()
            except Exception as e:
                logging.error(f"6.5 wandb.finish() {e}")

    except Exception as e:
        logging.error(f" 7 Errore durante l'addestramento di {run_name}: {e}")
        traceback.print_exc()

        wandb.finish()
        raise e

def main():
    """Funzione principale per eseguire l'addestramento."""
    parser = argparse.ArgumentParser(description='Script per l\'addestramento di YOLOv10 con Ultralytics')
    parser.add_argument('--config', type=str, required=True, help='Percorso al file YAML di configurazione del modello')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Percorso al file YAML di configurazione del dataset')

    args = parser.parse_args()

    model_config_path = Path(args.config)
    if not model_config_path.is_absolute():
        model_config_path = BASE_DIR / model_config_path

    dataset_config_path = Path(args.dataset)
    if not dataset_config_path.is_absolute():
        dataset_config_path = BASE_DIR / dataset_config_path

    if not model_config_path.exists():
        logging.error(f"File di configurazione del modello non trovato: {model_config_path}")
        sys.exit(1)

    if not dataset_config_path.exists():
        logging.error(f"File di configurazione del dataset non trovato: {dataset_config_path}")
        sys.exit(1)

    try:
        logging.info(
            f"Inizio dell'addestramento per la configurazione: {model_config_path} con dataset: {dataset_config_path}")
        train_model(model_config_path, dataset_config_path)
        logging.info(f"Addestramento completato per: {model_config_path}")
    except Exception as e:
        logging.error(f"Addestramento interrotto per {model_config_path}: {e}")


if __name__ == "__main__":
    main()
