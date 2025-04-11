# Import all components to trigger registration



from .ingestion.contract import ContractIngestor
from .ingestion.email import EmailIngestor
from .ingestion.deal import DealIngestor
from .ingestion.recap import RecapIngestor

from .cleaning.contract import ContractCleaner
from .cleaning.email import EmailCleaner
from .cleaning.deal import DealCleaner
from .cleaning.recap import RecapCleaner

from .chunking.contract import ContractChunker
from .chunking.email import EmailChunker
from .chunking.deal import DealChunker