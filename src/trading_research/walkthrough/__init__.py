"""The walkthrough: the pipeline's steps on one dataset, each choice made by the pipeline itself.

A reader follows it one command at a time (``trp search ...``), from Python
(:class:`trading_research.api.Pipeline`), or in one go on generated data
(``trp demo``):

:mod:`.layout`
    the dataset's layout, the views that hide the held-out days from every
    step before the read, and a synthetic market written in that layout;
:mod:`.record`
    a search's run directory: its registration, one folder per step, and the
    ledger of its one held-out read;
:mod:`.signal`
    features, labels at a hold, a candidate scored by a walk-forward, the
    chosen signal fitted and turned into decisions;
:mod:`.execution`
    one signal as a taker, a passive entry and a market maker, at a stated fee;
:mod:`.verdict`
    the clustered statistic, the kills in order, the placebos, Holm;
:mod:`.steps`
    the six steps: instruments, frequency, method, model, execution, read;
:mod:`.demo` and :mod:`.agent`
    the demonstration, and the agent harness run from the command line.

Every computation is the repository's own (the screen, the stages, the
walk-forward and its evaluators, the executors, the verdict functions); this
package orders them and keeps their files.
"""
