"""Valuation methods: pure functions over the conformed panel (spec section 10).

Importing the package registers every family. The list lived in cli.py and in the tests
and drifted apart once already; one copy, next to the modules it names, cannot.
"""

import cryptoval.methods.benchmarks  # noqa: F401 -- import registers the family
import cryptoval.methods.cashflow  # noqa: F401
import cryptoval.methods.cohort  # noqa: F401
import cryptoval.methods.cost  # noqa: F401
import cryptoval.methods.monetary  # noqa: F401
import cryptoval.methods.multiples  # noqa: F401
import cryptoval.methods.network  # noqa: F401
