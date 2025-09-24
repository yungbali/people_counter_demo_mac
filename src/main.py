"""
Main entry point for the occupancy security system.
"""

import asyncio
import signal
import sys
from typing import Optional

from src.config.manager import config_manager
from src.utils.logging import setup_logging, get_logger


class OccupancySecuritySystem:
    """Main system orchestrator."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.running = False
        self.config = None
    
    async def initialize(self) -> None:
        """Initialize the system components."""
        try:
            # Load configuration
            self.config = config_manager.load_config()
            self.logger.info(f"Loaded configuration for site: {self.config.site.name}")
            
            # Validate configuration
            errors = config_manager.validate_config()
            if errors:
                self.logger.error("Configuration validation errors:")
                for error in errors:
                    self.logger.error(f"  - {error}")
                raise ValueError("Invalid configuration")
            
            self.logger.info("System initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            raise
    
    async def start(self) -> None:
        """Start the system services."""
        self.logger.info("Starting occupancy security system...")
        self.running = True
        
        # TODO: Initialize and start services in subsequent tasks
        # - CV Service
        # - Aggregation Service  
        # - Alert Service
        # - API Service
        
        self.logger.info("System started successfully")
    
    async def stop(self) -> None:
        """Stop the system services."""
        self.logger.info("Stopping occupancy security system...")
        self.running = False
        
        # TODO: Stop services gracefully
        
        self.logger.info("System stopped")
    
    async def run(self) -> None:
        """Main system run loop."""
        await self.initialize()
        await self.start()
        
        try:
            # Keep running until stopped
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        finally:
            await self.stop()


def signal_handler(system: OccupancySecuritySystem):
    """Handle shutdown signals."""
    def handler(signum, frame):
        asyncio.create_task(system.stop())
    return handler


async def main():
    """Main entry point."""
    # Set up logging
    setup_logging(log_level="INFO", log_file="logs/system.log")
    logger = get_logger(__name__)
    
    logger.info("Starting Occupancy Security System")
    
    # Create system instance
    system = OccupancySecuritySystem()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler(system))
    signal.signal(signal.SIGTERM, signal_handler(system))
    
    try:
        await system.run()
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())