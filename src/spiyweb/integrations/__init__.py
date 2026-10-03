"""Adapters that put a Spiyweb index behind other frameworks' retriever APIs.

Each adapter lives in its own module and needs its own extra -
`spiyweb[langchain]` for `spiyweb.integrations.langchain`,
`spiyweb[llamaindex]` for `spiyweb.integrations.llamaindex`. Nothing here is
imported by `import spiyweb`: the frameworks are optional, never a
dependency, exactly like every other heavy library this project touches.

The adapters add no retrieval logic. They call `SpiywebIndex.retrieve`, keep
the web's own ranking (accumulated energy, not cosine), and carry what makes
the web different from a top-k list - votes, hop distance, the source -
into each framework's metadata, where a caller's chain can use it.
"""
