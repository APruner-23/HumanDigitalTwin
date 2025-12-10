"""
IoT data generator based on the onto.owl ontology.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import random
from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS


class OntologyDataGenerator:
    """
    Generates realistic IoT data based on the ontology.
    Analyzes onto.owl to understand which sensors/devices exist and generates compliant data.
    """

    def __init__(self, ontology_path: str = None):
        """
        Initialize the generator.

        Args:
            ontology_path: Path to the onto.owl file
        """
        if ontology_path is None:
            # Default path
            project_root = Path(__file__).parent.parent.parent
            ontology_path = project_root / "ontologies" / "onto.owl"

        self.ontology_path = Path(ontology_path)
        self.graph = Graph()

        # Load ontology
        if self.ontology_path.exists():
            self.graph.parse(str(self.ontology_path))

        # Ontology namespace
        self.ns = Namespace("http://www.semanticweb.org/jbagwell/ontologies/2017/9/untitled-ontology-6#")

        # Define device profiles based on ontology
        self._init_device_profiles()

    def _init_device_profiles(self):
        """Initializes device profiles based on the ontology."""

        # Fitbit - focus on activity tracking and sleep
        self.device_profiles = {
            "fitbit": {
                "device_type": "fitbit",
                "sensors": ["heartRateMonitor", "3axisAccelerometer", "altimeter", "gps"],
                "metrics": {
                    # Steps and activity
                    "steps": (0, 20000),
                    "minutesSedentary": (200, 800),
                    "minutesLightlyActive": (50, 200),
                    "minutesFairlyActive": (10, 60),
                    "minutesVeryActive": (0, 60),
                    "floors": (0, 30),
                    "distanceMiles": (0.0, 15.0),
                    "elevation": (0, 500),
                    "calories": (1500, 3500),

                    # Sleep
                    "minutesAsleep": (300, 540),
                    "minutesAwake": (5, 60),
                    "awakeningsCount": (0, 10),
                    "minutesToFallAsleep": (5, 30),
                    "minutesAfterWakeup": (5, 30),
                    "timeInBed": (360, 600),

                    # User data
                    "weight": (50.0, 120.0),
                    "height": (150.0, 200.0),
                    "bmi": (18.0, 35.0),
                    "fat": (10.0, 40.0),
                }
            },
            "garmin": {
                "device_type": "garmin",
                "sensors": ["heartRateMonitor", "gps", "altimeter"],
                "metrics": {
                    # Heart rate
                    "averageHeartRate": (60, 150),
                    "maximunHeartRate": (120, 200),

                    # Activity
                    "activityName": None,  # Stringa
                    "activityType": None,  # Stringa
                    "duration": (300, 7200),  # secondi
                    "distanceMiles": (0.0, 26.0),
                    "calories": (100, 2000),
                    "averageSpeedPace": (5.0, 15.0),  # mph
                    "maxSpeedBestPace": (8.0, 20.0),

                    # Advanced metrics
                    "efficiency": (0.5, 1.0),
                    "estamatedIntensityFactor": (0.5, 1.2),
                    "estimatedTrainingStessScore": (0, 300),
                }
            },
            "jawbone": {
                "device_type": "jawbone",
                "sensors": ["3axisAccelerometer", "3axisGyroscope"],
                "metrics": {
                    # Activity
                    "steps": (0, 20000),
                    "activeTime": (0, 300),  # minuti
                    "activeTimeSeconds": (0, 18000),
                    "distanceKilometers": (0.0, 20.0),
                    "calories": (1500, 3500),
                    "inactive": (300, 1200),  # sedentary minutes
                    "percentActive": (10.0, 80.0),
                }
            }
        }

        # Activity types for Garmin
        self.activity_types = [
            "running", "cycling", "walking", "swimming", "gym",
            "hiking", "yoga", "tennis", "soccer"
        ]

    def generate_data(
        self,
        device_type: str,
        device_id: Optional[str] = None,
        num_records: int = 1,
        time_interval_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Generates IoT data for a specific device.

        Args:
            device_type: Device type ('fitbit', 'garmin', 'jawbone')
            device_id: Device ID (if None, generates automatically)
            num_records: Number of records to generate
            time_interval_minutes: Interval between records in minutes

        Returns:
            List of dictionaries with generated data

        Raises:
            ValueError: If device_type is not supported
        """
        if device_type not in self.device_profiles:
            available = ', '.join(self.device_profiles.keys())
            raise ValueError(f"Device type '{device_type}' non supportato. Disponibili: {available}")

        profile = self.device_profiles[device_type]

        if device_id is None:
            device_id = f"{device_type}_{random.randint(1000, 9999)}"

        records = []
        base_time = datetime.now()

        for i in range(num_records):
            timestamp = base_time - timedelta(minutes=i * time_interval_minutes)

            # Generate data for all metrics
            data = {}
            for metric, value_range in profile["metrics"].items():
                if value_range is None:
                    # String metrics (e.g. activityType)
                    if metric == "activityType":
                        data[metric] = random.choice(self.activity_types)
                    elif metric == "activityName":
                        data[metric] = f"Activity_{random.randint(1, 100)}"
                else:
                    # Numeric metrics
                    min_val, max_val = value_range
                    if isinstance(min_val, float):
                        data[metric] = round(random.uniform(min_val, max_val), 2)
                    else:
                        data[metric] = random.randint(min_val, max_val)

            record = {
                "device_type": device_type,
                "device_id": device_id,
                "timestamp": timestamp.isoformat(),
                "data": data,
                "metadata": {
                    "sensors": profile["sensors"],
                    "generated": True,
                    "ontology": "onto.owl"
                }
            }

            records.append(record)

        return records

    def get_available_devices(self) -> List[str]:
        """
        Returns the list of available devices.

        Returns:
            List of device names
        """
        return list(self.device_profiles.keys())

    def get_device_metrics(self, device_type: str) -> Dict[str, Any]:
        """
        Returns available metrics for a device.

        Args:
            device_type: Device type

        Returns:
            Dictionary with metrics and their ranges

        Raises:
            ValueError: If device_type does not exist
        """
        if device_type not in self.device_profiles:
            raise ValueError(f"Device type '{device_type}' non trovato")

        profile = self.device_profiles[device_type]
        return {
            "device_type": device_type,
            "sensors": profile["sensors"],
            "metrics": profile["metrics"]
        }

    def generate_realistic_day(self, device_type: str, device_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generates a full day of realistic data (24 samples, one per hour).

        Args:
            device_type: Device type
            device_id: Device ID

        Returns:
            List of 24 records, one for each hour
        """
        return self.generate_data(
            device_type=device_type,
            device_id=device_id,
            num_records=24,
            time_interval_minutes=60
        )

    def generate_sample_for_all_devices(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generates sample data for all available devices.

        Returns:
            Dictionary with device_type as key and list of records as value
        """
        samples = {}
        for device_type in self.device_profiles.keys():
            samples[device_type] = self.generate_data(
                device_type=device_type,
                num_records=5
            )
        return samples
