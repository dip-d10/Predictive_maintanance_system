import os
import sys
import pandas as pd
from datetime import datetime

from src.entity.config_entity import AlertConfig
from src.entity.artifact_entity import AlertArtifact
from src.exception import MyException
from src.logger import logging


class AlertManager:
    """
    Trigger alerts when maintenance is required.
    Currently implements logging/mock alerts.
    Can be extended to integrate with email, Slack, etc.
    """
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.alerts_log_dir = config.alerts_log_dir
        os.makedirs(self.alerts_log_dir, exist_ok=True)
    
    def _log_alert(self, machine_id: str, failure_probability: float, risk_level: str):
        """Log alert to file and console"""
        alert_message = (
            f"🚨 MAINTENANCE ALERT | "
            f"Machine: {machine_id} | "
            f"Risk: {risk_level} | "
            f"Failure Prob: {failure_probability:.4f}"
        )
        logging.warning(alert_message)
        return alert_message
    
    def _save_alert_to_file(self, alerts: list) -> str:
        """Save alerts to file for audit trail"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            alert_file = os.path.join(
                self.alerts_log_dir,
                f"alerts_{timestamp}.log"
            )
            
            with open(alert_file, "w") as f:
                for alert in alerts:
                    f.write(f"{alert}\n")
            
            logging.info(f"Alerts saved to {alert_file}")
            return alert_file
            
        except Exception as e:
            logging.warning(f"Could not save alerts to file: {e}")
            return ""
    
    def trigger_alerts(self, predictions_df: pd.DataFrame) -> list:
        """
        Trigger alerts for machines requiring maintenance
        """
        try:
            alerts = []
            
            # Filter predictions requiring maintenance
            maintenance_required = predictions_df[predictions_df['maintenance_required']]
            
            if len(maintenance_required) == 0:
                logging.info("No maintenance alerts triggered")
                return alerts
            
            logging.info(f"Triggering {len(maintenance_required)} maintenance alerts")
            
            # Sort by risk level (HIGH first)
            risk_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
            maintenance_required = maintenance_required.copy()
            maintenance_required['risk_sort'] = maintenance_required['risk_level'].map(risk_order)
            maintenance_required = maintenance_required.sort_values('risk_sort')
            
            # Trigger alerts
            for _, row in maintenance_required.iterrows():
                alert_msg = self._log_alert(
                    row['machineID'],
                    row['failure_probability'],
                    row['risk_level']
                )
                alerts.append({
                    'machine_id': row['machineID'],
                    'timestamp': row['datetime'],
                    'failure_probability': row['failure_probability'],
                    'risk_level': row['risk_level'],
                    'alert_message': alert_msg,
                    'alert_time': datetime.utcnow().isoformat()
                })
            
            # Save alerts
            alert_messages = [alert['alert_message'] for alert in alerts]
            self._save_alert_to_file(alert_messages)
            
            return alerts
            
        except Exception as e:
            logging.exception(f"Error triggering alerts: {e}")
            raise MyException(e, sys)
    
    def _mock_email_notification(self, alerts: list) -> bool:
        """
        Mock email notification (placeholder for actual integration)
        """
        try:
            if not self.config.enable_email:
                return False
            
            high_risk_alerts = [a for a in alerts if a['risk_level'] == 'HIGH']
            
            if high_risk_alerts:
                logging.info(
                    f"[MOCK EMAIL] Sending {len(high_risk_alerts)} HIGH risk alerts to {self.config.email_recipients}"
                )
            
            return True
            
        except Exception as e:
            logging.warning(f"Mock email notification failed: {e}")
            return False
    
    def _mock_slack_notification(self, alerts: list) -> bool:
        """
        Mock Slack notification (placeholder for actual integration)
        """
        try:
            if not self.config.enable_slack:
                return False
            
            if alerts:
                logging.info(
                    f"[MOCK SLACK] Posting {len(alerts)} alerts to {self.config.slack_channel}"
                )
            
            return True
            
        except Exception as e:
            logging.warning(f"Mock Slack notification failed: {e}")
            return False
    
    def initiate_alert_manager(self, predictions_df: pd.DataFrame) -> AlertArtifact:
        """
        Main entry point for alert management
        """
        try:
            logging.info("Starting alert management")
            
            # Trigger alerts
            alerts = self.trigger_alerts(predictions_df)
            
            # Send mock notifications
            email_sent = self._mock_email_notification(alerts)
            slack_sent = self._mock_slack_notification(alerts)
            
            artifact = AlertArtifact(
                alerts_triggered=len(alerts),
                high_risk_alerts=sum(1 for a in alerts if a['risk_level'] == 'HIGH'),
                medium_risk_alerts=sum(1 for a in alerts if a['risk_level'] == 'MEDIUM'),
                alerts_list=alerts,
                email_notification_sent=email_sent,
                slack_notification_sent=slack_sent,
                alert_timestamp=datetime.utcnow().isoformat(),
                is_alert_successful=True,
                message=f"Alert manager completed. {len(alerts)} alerts triggered."
            )
            
            logging.info("Alert management completed")
            return artifact
            
        except Exception as e:
            logging.exception(f"Error in alert manager: {e}")
            raise MyException(e, sys)
