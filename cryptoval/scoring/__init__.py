"""Walk-forward scoring (spec 11.1-11.3, 11.6).

Importing the package registers every model family. The method layer learned this the
hard way -- a registration list kept in two places drifts -- so there is one copy here,
next to the modules it names.
"""

import cryptoval.scoring.baselines  # noqa: F401 -- import registers the family
# import cryptoval.scoring.linear      -- restored in Task 7
# import cryptoval.scoring.nonlinear   -- restored in Task 8
