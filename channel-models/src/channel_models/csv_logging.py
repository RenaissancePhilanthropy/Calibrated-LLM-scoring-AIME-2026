import atexit
import contextlib
import csv
import logging
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

CsvValue = str | int | float | None

logger = logging.getLogger(__name__)


class AsyncCSVLogger:
    """High-performance async CSV logger with batching and minimal overhead."""

    def __init__(self, batch_size: int = 100, flush_interval: float = 5.0) -> None:
        self._batch_size = batch_size
        self._flush_interval = flush_interval

        # Per-file data structures
        self._row_batches: dict[str, list[list[str | int | float]]] = {}
        self._stored_headers: dict[str, list[str]] = {}
        self._last_flush_times: dict[str, float] = {}

        # Threading
        self._queue: queue.Queue[Callable[[], None] | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
        atexit.register(self.shutdown)

    def _format_value(self, value: CsvValue) -> str | int | float:
        """Format a single value for CSV writing, preserving types for CSV quoting."""
        if value is None:
            return ""
        if isinstance(value, float):
            # Round floats to 4 decimal places but keep as float for QUOTE_NONNUMERIC
            return round(value, 4)
        # Keep strings as strings, numbers as numbers for proper CSV quoting
        return value

    def _format_row(self, row_data: dict[str, CsvValue]) -> list[str | int | float]:
        """Pre-format a row of data for efficient CSV writing."""
        return [self._format_value(value) for value in row_data.values()]

    def _write_batch_to_file(
        self,
        rows: list[list[str | int | float]],
        filename: str,
        headers: list[str] | None = None,
    ) -> None:
        """Write accumulated rows for a file to disk."""
        file_exists = Path(filename).exists()
        file_needs_headers = not file_exists

        # Check if existing file needs headers
        if file_exists and headers:
            try:
                with Path(filename).open(newline="", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    # If first line doesn't look like headers (no alphabetic chars
                    # in first field), we need to add headers
                    if first_line and not any(
                        c.isalpha() for c in first_line.split(",")[0]
                    ):
                        file_needs_headers = True
            except OSError:
                # If we can't read the file, assume it needs headers
                file_needs_headers = True

        try:
            # If file needs headers, we need to read existing content and rewrite
            if file_exists and file_needs_headers:
                # Read existing content
                with Path(filename).open(newline="", encoding="utf-8") as f:
                    existing_content = f.read()

                # Write new file with headers + existing content + new rows
                with Path(filename).open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
                    if headers is not None:
                        writer.writerow(headers)
                    f.write(existing_content)
                    writer.writerows(rows)
            else:
                # Normal case: append to existing file or create new file
                mode = "a" if file_exists else "w"
                with Path(filename).open(mode, newline="", encoding="utf-8") as f:
                    writer = csv.writer(
                        f, quoting=csv.QUOTE_NONNUMERIC
                    )  # Quote all non-numeric values

                    # Write headers if headers are needed and available
                    if file_needs_headers and headers:
                        writer.writerow(headers)

                    # Write all batched rows at once
                    writer.writerows(rows)
        except OSError:
            logger.exception("Error writing CSV batch to %s", filename)

    def _writer_loop(self) -> None:
        """Background thread that processes write operations and periodic flushes."""
        while not self._stop_event.is_set():
            try:
                # Process any queued flush operations
                try:
                    item = self._queue.get(timeout=0.1)
                    if item is None:  # Shutdown signal
                        break
                    item()
                    self._queue.task_done()
                except queue.Empty:
                    pass

            except Exception:  # intentional broad catch in daemon thread
                logger.exception("Error in CSV writer loop")

    def queue_csv_row(
        self,
        filename: str,
        row_data: dict[str, CsvValue],
        headers: list[str] | None = None,
    ) -> None:
        """Queue a single row for batched writing to a CSV file."""
        if self._stop_event.is_set():
            return

        # Initialize batch for this file if needed
        if filename not in self._row_batches:
            self._row_batches[filename] = []
            self._last_flush_times[filename] = time.time()

        # Store headers for this file if provided (update if new headers given)
        if headers:
            self._stored_headers[filename] = headers

        # Pre-format the row data
        formatted_row = self._format_row(row_data)
        self._row_batches[filename].append(formatted_row)

        # If batch is full, queue a flush operation
        if len(self._row_batches[filename]) >= self._batch_size:
            self.flush_file(filename, headers=headers)

        # Check for files that need periodic flushing
        current_time = time.time()
        for file_key in list(self._row_batches.keys()):
            if (
                file_key in self._last_flush_times
                and current_time - self._last_flush_times[file_key]
                >= self._flush_interval
            ):
                self.flush_file(file_key)

    def flush_file(self, filename: str, headers: list[str] | None = None) -> None:
        """Force flush of all pending rows for a specific file."""
        if self._stop_event.is_set():
            return
        if filename not in self._row_batches:
            logger.debug("Attempting to flush untracked file '%s'.", filename)
            return
        rows = self._row_batches[filename]
        if len(rows) == 0:
            logger.debug("No rows to flush in file '%s'.", filename)
            return
        self._row_batches[filename] = []
        # If headers weren't provided but we have them stored, use stored headers
        if not headers:
            headers = self._stored_headers.get(filename)
        logger.debug("Queuing %d rows to '%s'.", len(rows), filename)

        def write_batch(
            r: list[list[str | int | float]] = rows,
            f: str = filename,
            h: list[str] | None = headers,
        ) -> None:
            self._write_batch_to_file(r, f, h)

        self._queue.put(write_batch)
        self._last_flush_times[filename] = time.time()

    def flush_all(self) -> None:
        """Force flush of all pending rows for all files."""
        for filename in list(self._row_batches.keys()):
            self.flush_file(filename)

    def queue_write(self, write_func: Callable[[], None]) -> None:
        """Queue a custom write function (for backward compatibility)."""
        if not self._stop_event.is_set():
            self._queue.put(write_func)

    def shutdown(self, shutdown_timeout: float = 5.0) -> None:
        """Shutdown the logger, ensuring all writes complete."""
        if self._writer_thread and self._writer_thread.is_alive():
            self.flush_all()
            self._queue.put(
                None
            )  # poison; will shutdown the writer thread when processed
            self._writer_thread.join(timeout=shutdown_timeout)
            self._stop_event.set()

    def repair_csv_headers(self, filename: str, headers: list[str]) -> bool:
        """Repair a CSV file that is missing headers by adding them at the beginning."""
        if not Path(filename).exists():
            logger.warning("File %s does not exist, cannot repair headers", filename)
            return False

        temp_filename = filename + ".tmp"
        try:
            # Read the existing content
            with Path(filename).open(newline="", encoding="utf-8") as f:
                content = f.read()

            # Check if file already has headers
            first_line = content.split("\n")[0].strip()
            if first_line and any(c.isalpha() for c in first_line.split(",")[0]):
                logger.info("File %s already has headers, no repair needed", filename)
                return True

            # Create a temporary file with headers
            with Path(temp_filename).open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow(headers)
                f.write(content)

            # Replace the original file
            Path(temp_filename).replace(filename)
            logger.info("Successfully repaired headers for %s", filename)

            # Update our tracking
            self._stored_headers[filename] = headers

        except (OSError, UnicodeDecodeError, csv.Error):
            logger.exception("Error repairing headers for %s", filename)
            with contextlib.suppress(OSError):
                Path(temp_filename).unlink(missing_ok=True)
            return False
        else:
            return True

    def add_headers_to_file(self, filename: str, headers: list[str]) -> bool:
        """Explicitly add headers to an existing file by rewriting it."""
        if not Path(filename).exists():
            logger.warning("File %s does not exist, cannot add headers", filename)
            return False

        temp_filename = filename + ".tmp"
        try:
            # Read the existing content
            with Path(filename).open(newline="", encoding="utf-8") as f:
                content = f.read()

            # Check if file already has headers
            first_line = content.split("\n")[0].strip()
            if first_line and any(c.isalpha() for c in first_line.split(",")[0]):
                logger.info("File %s already has headers, no change needed", filename)
                return True

            # Create a temporary file with headers
            with Path(temp_filename).open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow(headers)
                f.write(content)

            # Replace the original file
            Path(temp_filename).replace(filename)
            logger.info("Successfully added headers to %s", filename)

            # Update our tracking
            self._stored_headers[filename] = headers

        except (OSError, UnicodeDecodeError, csv.Error):
            logger.exception("Error adding headers to %s", filename)
            with contextlib.suppress(OSError):
                Path(temp_filename).unlink(missing_ok=True)
            return False
        else:
            return True
