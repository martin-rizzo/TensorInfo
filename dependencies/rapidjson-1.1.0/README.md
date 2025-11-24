# RapidJSON

**RapidJSON** is a fast JSON parser/generator for C++ with both SAX/DOM style API. This version is a stripped-down version that includes only the headers necessary to build with a modern compilers. The BSD3 code from the 'include/rapidjson/msinttypes' directory, which provided support for Visual Studio 2013 and earlier, was removed.

## Why RapidJSON?

RapidJSON was chosen for its exceptional performance. It offers high parsing and generation speeds while maintaining a small footprint, making it ideal for applications where efficiency is crucial.

Benchmark tests comparing various open-source JSON parsers and generators are available [here](https://github.com/miloyip/nativejson-benchmark).

## Code Modifications

A line in "document.h" was temporarily commented out due to a compiler error and unclear intent, allowing development to proceed without delay. The specific issue involved an attempt to modify the read-only member, as indicated by the error:
 
```
document.h:319:82: error: assignment of read-only member ‘rapidjson::GenericStringRef<CharType>::length’
```

Further investigation is planned to resolve this issue.

## License

RapidJSON is licensed under the MIT license.\
Details can be found in the [LICENSE](LICENSE) file within this directory.\
For more information or to access the full version, visit **[RapidJSON repository on GitHub](https://github.com/Tencent/rapidjson)**.\
I would like to thank Tencent for supporting the open source community by making RapidJSON available to the public.
