v3.7.3 / 2016-12-04
==================

  * Update `winston-daily-rotate-file`@1.4.0

v3.7.2 / 2016-11-06
==================

  * Tested against `node`@7
  * Remove support for `node`@0
  * Update `winston`@2.3.0

v3.7.1 / 2016-09-12
==================

  * Update `basic-authentication`@1.7.0
  * Update `transfer-rate`@1.2.0
  * Update `winston-daily-rotate-file`@1.3.1

v3.7.0 / 2016-09-03
==================

  * Rotate log each day
  * Add `daily` (options)
  * Tested against `node`@6

v3.6.1 / 2016-03-06
==================

  * Tested against `node`@5.3
  * Update `winston`@2.2.0

v3.6.0 / 2015-12-04
==================

  * Tested against `node`@5
  * Update `winston`@2.1.1
  * Update `basic-authentication`@1.6.2
  * Update `transfer-rate`@1.1.3

v3.5.0 / 2015-10-26
==================

  * Tested against `node`@4
  * Update `winston`@1.1.1
  * Update `basic-authentication`@1.6.1
  * Update `transfer-rate`@1.1.2

v3.4.0 / 2015-08-27
==================

  * Tested against `iojs`@3
  * Small improvements
  * Update `basic-authentication`@1.6.0
  * Update `transfer-rate`@1.1.1

v3.3.5 / 2015-06-30
==================

  * Update `winston`@1.0.1

v3.3.4 / 2015-06-07
==================

  * Fix SPDX license
  * Update `on-finished`@2.2.1

v3.3.3 / 2015-05-05
==================

  * Update `on-finished`@2.2.1

v3.3.2 / 2015-04-11
==================

  * Update `transfer-rate`@1.1.0
  * Update `winston`@1.0.0

v3.3.1 / 2015-02-28
==================

  * Ignore `/coverage` for npm
  * Update `basic-authentication`@1.5.12

v3.3.0 / 2015-02-15
==================

  * Add `transports` (options)
  * `filename` (options) default value is `false`
  * `coveralls` test

v3.2.8 / 2015-02-05
==================

  * `windows` test
  * `iojs` test
  * Update `basic-authentication`@1.5.11
  * Update `transfer-rate`@1.0.6
  * Update `winston`@0.9.0

v3.2.7 / 2015-01-02
==================

  * Update `transfer-rate`@1.0.5

v3.2.6 / 2014-12-30
==================

  * Remove "x-forwarded-for" ip

v3.2.5 / 2014-12-23
==================

  * Use `hrtime` before/after all
  * Update `on-finished`@2.2.0

v3.2.4 / 2014-12-19
==================

  * Decrease decimal unit to .2 in `response`

v3.2.3 / 2014-12-02
==================

  * Callback function inside "custom" object

v3.2.2 / 2014-11-21
==================

  * Update `basic-authentication`@1.5.9

v3.2.1 / 2014-11-15
==================

  * Small improvement

v3.2.0 / 2014-11-10
==================

  * Add `transfer` (options)
  * Using `transfer-rate`@1.0.4

v3.1.2 / 2014-11-06
==================

  * Fix for Node 0.11
  * Update `winston`@0.8.3

  v3.1.1 / 2014-11-03
==================

  * Strict If

v3.1.0 / 2014-11-01
==================

  * Don't check `next` callback
  * Add `function` (options)

v3.0.13 / 2014-10-26
==================

  * Update `basic-authentication`@1.5.8

v3.0.12 / 2014-10-26
==================

  * `jshint`

v3.0.11 / 2014-10-23
==================

  * Update `on-finished`@2.1.1

v3.0.10 / 2014-10-08
==================

  * Update `winston`@0.8.1

v3.0.9 / 2014-10-04
==================

  * Remove jsdoc
  * `package.json` min

v3.0.8 / 2014-09-29
==================

  * Little fix

v3.0.7 / 2014-09-18
==================

  * Update `winston`@0.8.0

v3.0.5 / 2014-09-01
==================

  * New `grunt-endline`@0.1.0

v3.0.4 / 2014-09-01
==================

  * Update `basic-authentication`@1.5.5

v3.0.3 / 2014-08-23
==================

  * Performance tips
  * Update `basic-authentication`@1.5.4

v3.0.2 / 2014-08-17
==================

  * Change `req` to `res` on-finished callback
  * Update `on-finished`@2.1.0

v3.0.1 / 2014-08-16
==================

  * Save IP for multiple chunks
  * Using `on-finished`@2.0.0

v3.0.0 / 2014-08-14
==================

  * "deprecated" (options)
  * Using `finished`@1.2.2
  * Use event listener instead of rewrite `res.end`
   * You can use old method with "deprecated" flag

v2.2.5 / 2014-08-11
==================

  * Update README.md
  * Update `basic-authentication`@1.5.2

v2.2.3 / 2014-08-04
==================

  * Doc update

v2.2.2 / 2014-07-31
==================

  * Fix "bytesRes" and "bytesReq"
  * Add "headers" (options)

v2.2.0 / 2014-07-27
==================

  * Test "node": ">=0.10.0" only
  * Using task runner `grunt`
  * Using test framework `mocha`
  * Testing script will be put inside "test/"
  * ".npmignore" more aggressive
  * `uglify` compiles
  * `jsdoc` documentation

v2.1.1 / 2014-07-25
==================

  * Update `basic-authentication`@1.4.0

v2.1.0 / 2014-07-25
==================

  * Rewrite for multiple require

v2.0.3 / 2014-07-24
==================

  * Fix `options.custom` if void

v2.0.2 / 2014-07-23
==================

  * `res.end` correct return Boolean
  * Fix `options.winston.silent` for display only console out and not file

v2.0.0 / 2014-07-22
==================

  * Add `custom` object. Now can change log output
  * Change _API_. This setting now, are under `winston` object
   * `logger`
   * `level`
   * `silent`
   * `colorize`
   * `timestamp`
   * `maxsize`
   * `maxFiles`
   * `json`
   * `raw`

v1.3.0 / 2014-07-19
==================

  * _log_ Rename "byte" to "bytesRes", bytes read
  * _log_ New "bytesReq", bytes dispatched
  * _log_ New "__filename", who wrote log
  * Now working even as a function

v1.2.0 / 2014-07-17
==================

  * Don"t store "cookie" anymore
  * Minor var to function
  * Change MINOR version, due critical issue with node 0.11

v1.1.9 / 2014-07-16
==================

  * Improve performance (remove closure)
  * Write log, after sending all stuff to client

v1.1.8 / 2014-07-09
==================

  * Fix critical issue with node 0.10.29 (callback)

v1.1.7 / 2014-06-29
==================

  * Remove anonymous function
  * Update [`express`](https://github.com/visionmedia/express) @ 4.4.5

v1.1.5 / 2014-06-18
==================

  * Clean code

v1.1.4 / 2014-06-15
==================

  * Small improvements
  * Print output bytes

v1.1.3 / 2014-06-08
==================

  * Improve closures functions

v1.1.2 / 2014-06-08
==================

  * Various fixes

v1.1.1 / 2014-06-05
==================

  * Best callback function

v1.1.0 / 2014-05-27
==================

  * Fix status code
  * JsDoc improvement

v1.0.10 / 2014-05-25
==================

  * Fix "logger" arg

v1.0.9 / 2014-05-24
==================

  * Better use of memory
  * "raw" (options)

v1.0.8 / 2014-05-21
==================

  * Logger option related to [this](https://github.com/flatiron/winston#working-with-multiple-loggers-in-winston)

v1.0.7 / 2014-05-21
==================

  * Standalone options flag

v1.0.6 / 2014-05-21
==================

  * Validate options flag
  * Update Expressjs to 4.3.0

v1.0.5 / 2014-05-19
==================

  * Timestamp option not require a Boolean

v1.0.4 / 2014-05-16
==================

  * More Options, related to "winston" module

v1.0.3 / 2014-05-14
==================

  * Using nanosecond for respose time

v1.0.2 / 2014-05-13
==================

  * Fix filename option

v1.0.1 / 2014-05-13
==================

  * Fix

v1.0.0 / 2014-05-13
==================

  * Project start
