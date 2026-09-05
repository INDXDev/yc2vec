"""Source adapters.

Each adapter owns one upstream source and exposes the same shape, so new
sources are additive rather than surgery on the fetch stage. Adapters other
than ``yc_oss_api`` are disabled by default.
"""

from pipeline.adapters.base import Adapter, AdapterResult
from pipeline.adapters.company_website import CompanyWebsiteAdapter
from pipeline.adapters.yc_oss_api import YcOssApiAdapter

__all__ = ["Adapter", "AdapterResult", "CompanyWebsiteAdapter", "YcOssApiAdapter"]
