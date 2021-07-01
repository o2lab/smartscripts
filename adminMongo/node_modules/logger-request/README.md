# [logger-request](https://github.com/hex7c0/logger-request)

[![NPM version](https://img.shields.io/npm/v/logger-request.svg)](https://www.npmjs.com/package/logger-request)
[![Linux Status](https://img.shields.io/travis/hex7c0/logger-request.svg?label=linux-osx)](https://travis-ci.org/hex7c0/logger-request)
[![Windows Status](https://img.shields.io/appveyor/ci/hex7c0/logger-request.svg?label=windows)](https://ci.appveyor.com/project/hex7c0/logger-request)
[![Dependency Status](https://img.shields.io/david/hex7c0/logger-request.svg)](https://david-dm.org/hex7c0/logger-request)
[![Coveralls](https://img.shields.io/coveralls/hex7c0/logger-request.svg)](https://coveralls.io/r/hex7c0/logger-request)

HTTP request logger middleware for [nodejs](http://nodejs.org/), standalone logger and even more!

Save logs to file, show to console or both, to MongoDb, etc...

Look at [`logger-request-cli`](https://github.com/hex7c0/logger-request-cli) for Parser

## Installation

Install through NPM

```bash
npm install logger-request
```
or
```bash
git clone git://github.com/hex7c0/logger-request.git
```

## API

inside expressjs project
```js
var logger = require('logger-request');

var app = require('express')();

app.use(logger({
  filename: 'foo.log',
}));
```

### logger(options)

#### options

 - `transports` - **Array** Array of [winston transports](https://github.com/winstonjs/winston/blob/master/docs/transports.md) *(default "false")* 
 - `filename` - **String** If string, filename of the logfile to write output to *(default "false")*
 - `daily` - **Boolean** If true, rotate log each day *(default "false")*
 - `console` - **Boolean** If true, it displays log to console *(default "false")*
 - `standalone` - **Boolean** If true, return logger function instead of callback *(default "false")*
 - `deprecated` - **Boolean** Flag for write log after `res.end()`(true) instead of default `listener`(false) *(default "false")*
 - `functions` - **String** Use module like a function without `next` callback *(default "false")*
 - `winston` - **Object** Setting for selected transports
  - `logger` - **String** Logger option related to [`winston`](https://github.com/flatiron/winston#working-with-multiple-loggers-in-winston) *(default "logger-request")*
  - `level` - **String** Level of messages that this transport should log *(default "info")*
  - `silent` - **Boolean** Flag indicating whether to suppress output *(default "false")*
  - `colorize` - **Boolean** Flag indicating if we should colorize output *(default "false")*
  - `timestamp` - **Boolean|Function** Flag indicating if we should prepend output with timestamps *(default "true")*. If function is specified, its return value will be used instead of timestamps
  - `maxsize` - **Number** Max size in bytes of the logfile, if the size is exceeded then a new file is created *(default "8388608" [8Mb])*
  - `maxFiles` - **Number** Limit the number of files created when the size of the logfile is exceeded *(default "no limit")*
  - `json` - **Boolean** If true, messages will be logged as JSON *(default "true")*
  - `raw` - **Boolean** If true, raw messages will be logged to console *(default "false")*
  - `...` - **Mixed** Extra settings
 - `custom` - **Object** Setting for customization of logs
  - `pid` - **Boolean** Flag for `process.pid` *(default "disabled")*
  - `bytesReq` - **Boolean** Flag for `req.socket.bytesRead` *(default "disabled")*
  - `bytesRes` - **Boolean** Flag for `req.socket._bytesDispatched` *(default "disabled")*
  - `referer` - **Boolean** Flag for `req.headers['referer']` *(default "disabled")*
  - `auth` - **Boolean** Flag for [`basic-authentication`](https://github.com/hex7c0/basic-authentication) *(default "disabled")*
  - `transfer` - **Boolean** Flag for [`transfer-rate`](https://github.com/hex7c0/transfer-rate) *(default "disabled")*
  - `agent` - **Boolean** Flag for `req.headers['user-agent']` *(default "disabled")*
  - `lang` - **Boolean** Flag for `req.headers['accept-language']` *(default "disabled")*
  - `cookie` - **Boolean** Flag for `req.cookies` *(default "disabled")*
  - `headers` - **Boolean** Flag for `req.headers` *(default "disabled")*
  - `version` - **Boolean** Flag for `req.httpVersionMajor` *(default "disabled")*
  - `callback` - **Function** Flag for using callback function *(default "disabled")*

## Examples

Take a look at my [examples](examples)

### [License GPLv3](LICENSE)
