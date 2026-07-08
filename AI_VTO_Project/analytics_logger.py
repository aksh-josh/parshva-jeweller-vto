"""
analytics_logger.py — Real-time Analytics Logger for Power BI Dashboard
========================================================================
Background thread-based event logger that writes VTO analytics to MySQL
without blocking Flask request handlers.

FEATURES:
- Non-blocking queue-based event logging
- Background worker thread
- Connection pooling
- Session tracking
- Try-on event logging
- Accuracy metrics logging
- Gaze event logging

USAGE:
    logger = AnalyticsLogger(mysql_config)
    logger.start()
    
    # Log events (non-blocking)
    logger.log_event('session_start', session_id='abc123', user_id=1)
    logger.log_event('tryon', session_id='abc123', jewelry_id=42, action='add')
    
    # Cleanup on shutdown
    logger.stop()
"""

import threading
import queue
import time
from datetime import datetime
import mysql.connector
from contextlib import contextmanager


class AnalyticsLogger:
    def __init__(self, mysql_config):
        """
        Initialize analytics logger
        
        Args:
            mysql_config: dict with keys: host, user, password, database
        """
        self.mysql_config = mysql_config
        self.event_queue = queue.Queue(maxsize=1000)
        self.running = False
        self.worker_thread = None
        
        # Statistics
        self.events_logged = 0
        self.events_dropped = 0
        self.errors = 0
        
    def start(self):
        """Start background logging thread"""
        if self.running:
            print("[Analytics] Logger already running")
            return
            
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        print("[Analytics] Background logger started")
        
    def stop(self):
        """Stop background logging thread gracefully"""
        if not self.running:
            return
            
        print("[Analytics] Stopping logger...")
        self.running = False
        
        # Wait for queue to empty (max 5 seconds)
        timeout = time.time() + 5
        while not self.event_queue.empty() and time.time() < timeout:
            time.sleep(0.1)
        
        if self.worker_thread:
            self.worker_thread.join(timeout=2)
            
        print(f"[Analytics] Logger stopped. Stats: {self.events_logged} logged, "
              f"{self.events_dropped} dropped, {self.errors} errors")
        
    @contextmanager
    def _get_connection(self):
        """Context manager for MySQL connections with error handling"""
        conn = None
        try:
            conn = mysql.connector.connect(**self.mysql_config)
            yield conn
        except mysql.connector.Error as e:
            print(f"[Analytics] MySQL connection error: {e}")
            self.errors += 1
            raise
        finally:
            if conn and conn.is_connected():
                conn.close()
                
    def _process_queue(self):
        """Background worker thread - processes queued events"""
        print("[Analytics] Worker thread started")
        
        while self.running or not self.event_queue.empty():
            try:
                # Wait for event with timeout to allow clean shutdown
                event = self.event_queue.get(timeout=1)
                self._write_event(event)
                self.event_queue.task_done()
                self.events_logged += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Analytics] Queue processing error: {e}")
                self.errors += 1
                
        print("[Analytics] Worker thread stopped")
                
    def _write_event(self, event):
        """
        Write single event to database
        
        Args:
            event: dict with 'type' key and type-specific data
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # ═══════════════════════════════════════════════════════════
                # SESSION EVENTS
                # ═══════════════════════════════════════════════════════════
                
                if event['type'] == 'session_start':
                    cursor.execute("""
                        INSERT INTO vto_sessions 
                        (session_id, user_id, start_time, camera_type, gaze_enabled)
                        VALUES (%s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                        start_time = VALUES(start_time)
                    """, (
                        event['session_id'],
                        event.get('user_id'),
                        event['timestamp'],
                        event.get('camera_type', 'builtin'),
                        event.get('gaze_enabled', False)
                    ))
                    
                elif event['type'] == 'session_end':
                    cursor.execute("""
                        UPDATE vto_sessions 
                        SET end_time = %s, 
                            total_duration_sec = %s, 
                            total_frames_processed = %s
                        WHERE session_id = %s
                    """, (
                        event['timestamp'],
                        event.get('duration_sec', 0),
                        event.get('total_frames', 0),
                        event['session_id']
                    ))
                
                # ═══════════════════════════════════════════════════════════
                # TRY-ON EVENTS
                # ═══════════════════════════════════════════════════════════
                
                elif event['type'] == 'tryon':
                    cursor.execute("""
                        INSERT INTO vto_tryons 
                        (session_id, jewelry_id, jewelry_name, jewelry_category, 
                         action, trigger_method, timestamp, zoom_factor)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        event['session_id'],
                        event['jewelry_id'],
                        event.get('jewelry_name', 'Unknown'),
                        event.get('jewelry_category', 'unknown'),
                        event['action'],  # 'add' or 'remove'
                        event.get('trigger_method', 'click'),  # 'click', 'gaze', 'auto'
                        event['timestamp'],
                        event.get('zoom_factor', 1.0)
                    ))
                
                # ═══════════════════════════════════════════════════════════
                # ACCURACY METRICS
                # ═══════════════════════════════════════════════════════════
                
                elif event['type'] == 'accuracy':
                    cursor.execute("""
                        INSERT INTO vto_accuracy_metrics
                        (session_id, pck_5pct, mean_iou, mean_pos_error_px, fps,
                         detection_rate_pct, mean_jitter_px, overall_grade, 
                         quality_score, recorded_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        event['session_id'],
                        event.get('pck_5pct', 0.0),
                        event.get('mean_iou', 0.0),
                        event.get('mean_pos_error_px', 0.0),
                        event.get('fps', 0.0),
                        event.get('detection_rate_pct', 0.0),
                        event.get('mean_jitter_px', 0.0),
                        event.get('overall_grade', 'D'),
                        event.get('quality_score', 0.0),
                        event['timestamp']
                    ))
                
                # ═══════════════════════════════════════════════════════════
                # GAZE EVENTS
                # ═══════════════════════════════════════════════════════════
                
                elif event['type'] == 'gaze':
                    cursor.execute("""
                        INSERT INTO vto_gaze_events
                        (session_id, jewelry_id, gaze_x, gaze_y, dwell_duration_sec,
                         event_type, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        event['session_id'],
                        event.get('jewelry_id'),
                        event.get('gaze_x', 0.0),
                        event.get('gaze_y', 0.5),
                        event.get('dwell_duration_sec', 0.0),
                        event['event_type'],  # 'dwell_start', 'dwell_complete', 'add_to_tryon'
                        event['timestamp']
                    ))
                
                # ═══════════════════════════════════════════════════════════
                # UNKNOWN EVENT TYPE
                # ═══════════════════════════════════════════════════════════
                
                else:
                    print(f"[Analytics] Unknown event type: {event['type']}")
                    
                conn.commit()
                cursor.close()
                
        except Exception as e:
            print(f"[Analytics] Write error for {event.get('type', 'unknown')}: {e}")
            self.errors += 1
            
    def log_event(self, event_type, **kwargs):
        """
        Queue an event for background logging (non-blocking)
        
        Args:
            event_type: str - one of: 'session_start', 'session_end', 'tryon', 
                        'accuracy', 'gaze'
            **kwargs: Event-specific data
            
        Returns:
            bool: True if queued successfully, False if queue full
        """
        event = {
            'type': event_type,
            'timestamp': datetime.now(),
            **kwargs
        }
        
        try:
            self.event_queue.put_nowait(event)
            return True
        except queue.Full:
            self.events_dropped += 1
            print(f"[Analytics] Queue full, dropped {event_type} event")
            return False
            
    def get_stats(self):
        """Get logger statistics"""
        return {
            'running': self.running,
            'queue_size': self.event_queue.qsize(),
            'events_logged': self.events_logged,
            'events_dropped': self.events_dropped,
            'errors': self.errors,
            'success_rate': round(
                self.events_logged / max(self.events_logged + self.events_dropped, 1) * 100, 
                2
            )
        }


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTION: Session ID Generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_session_id():
    """Generate unique session ID using timestamp + random component"""
    import uuid
    return f"vto_{int(time.time())}_{str(uuid.uuid4())[:8]}"
