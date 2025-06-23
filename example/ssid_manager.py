import os
import json
import fcntl
from datetime import datetime, timedelta

class SSIDManager:
    LOCK_FILE = "ssid_locks.json"
    LOCK_TIMEOUT = 60  # seconds before considering a lock stale

    @staticmethod
    def _acquire_file_lock():
        """Create an exclusive file lock"""
        lock_file = open(f"{SSIDManager.LOCK_FILE}.lock", 'w')
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        return lock_file

    @staticmethod
    def _release_file_lock(lock_file):
        """Release the file lock"""
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()

    @staticmethod
    def _load_locks():
        """Load the current SSID locks"""
        if os.path.exists(SSIDManager.LOCK_FILE):
            try:
                with open(SSIDManager.LOCK_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _save_locks(locks):
        """Save the current SSID locks"""
        with open(SSIDManager.LOCK_FILE, 'w') as f:
            json.dump(locks, f)

    @staticmethod
    def _clean_stale_locks(locks):
        """Remove stale locks"""
        now = datetime.now().timestamp()
        return {
            ssid: data for ssid, data in locks.items()
            if now - data['timestamp'] < SSIDManager.LOCK_TIMEOUT
        }

    @staticmethod
    def acquire_ssid(demo_ssids):
        """
        Acquire an available demo SSID
        
        Args:
            demo_ssids (list): List of available demo SSIDs
            
        Returns:
            str: An available SSID or None if none available
        """
        lock_file = SSIDManager._acquire_file_lock()
        try:
            locks = SSIDManager._load_locks()
            locks = SSIDManager._clean_stale_locks(locks)
            
            # Find first available SSID
            for ssid in demo_ssids:
                if ssid not in locks:
                    locks[ssid] = {
                        'pid': os.getpid(),
                        'timestamp': datetime.now().timestamp()
                    }
                    SSIDManager._save_locks(locks)
                    return ssid
            
            return None
        finally:
            SSIDManager._release_file_lock(lock_file)

    @staticmethod
    def release_ssid(ssid):
        """
        Release a previously acquired SSID
        
        Args:
            ssid (str): The SSID to release
        """
        lock_file = SSIDManager._acquire_file_lock()
        try:
            locks = SSIDManager._load_locks()
            if ssid in locks and locks[ssid]['pid'] == os.getpid():
                del locks[ssid]
                SSIDManager._save_locks(locks)
        finally:
            SSIDManager._release_file_lock(lock_file)

    @staticmethod
    def update_lock(ssid):
        """
        Update the timestamp of a locked SSID
        
        Args:
            ssid (str): The SSID to update
        """
        lock_file = SSIDManager._acquire_file_lock()
        try:
            locks = SSIDManager._load_locks()
            if ssid in locks and locks[ssid]['pid'] == os.getpid():
                locks[ssid]['timestamp'] = datetime.now().timestamp()
                SSIDManager._save_locks(locks)
        finally:
            SSIDManager._release_file_lock(lock_file)