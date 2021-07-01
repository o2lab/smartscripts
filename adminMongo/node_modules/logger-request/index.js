'use strict';
/**
 * @file logger-request main
 * @module logger-request
 * @subpackage main
 * @version 3.7.0
 * @author hex7c0 <hex7c0@gmail.com>
 * @copyright hex7c0 2014
 * @license GPLv3
 */

/*
 * functions
 */
/**
 * output builder
 * 
 * @private
 * @function info
 * @param {Object} my - user options
 * @return {Function}
 */
function __info(my) {

  var promise = Array.call();
  if (my.pid) {
    promise.push([ 'pid', function() {

      return process.pid;
    } ]);
  }
  if (my.bytesReq) {
    promise.push([ 'bytesReq', function(req) {

      return req.socket.bytesRead;
    } ]);
  }
  if (my.bytesRes) {
    promise.push([ 'bytesRes', function(req) {

      return req.socket._bytesDispatched;
    } ]);
  }
  if (my.referer) {
    promise.push([ 'referer', function(req) {

      return req.headers.referer || req.headers.referrer;
    } ]);
  }
  if (my.auth) {
    var mod0 = require('basic-authentication')({
      legacy: true
    });
    promise.push([ 'auth', function(req) {

      return mod0(req).user;
    } ]);
  }
  if (my.transfer) {
    var mod1 = require('transfer-rate')();
    promise.push([ 'transfer', function(req) {

      return mod1(req, req.start);
    } ]);
  }
  if (my.agent) {
    promise.push([ 'agent', function(req) {

      return req.headers['user-agent'];
    } ]);
  }
  if (my.lang) {
    promise.push([ 'lang', function(req) {

      return req.headers['accept-language'];
    } ]);
  }
  if (my.cookie) {
    promise.push([ 'cookie', function(req) {

      return req.cookies;
    } ]);
  }
  if (my.headers) {
    promise.push([ 'headers', function(req) {

      return req.headers;
    } ]);
  }
  if (my.version) {
    promise.push([ 'version', function(req) {

      return req.httpVersionMajor + '.' + req.httpVersionMinor;
    } ]);
  }
  if (my.callback) {
    promise.push([ 'callback', function(req) {

      return my.callback(req);
    } ]);
  }
  var l = promise.length;

  if (l === 0) {
    /**
     * standard logger output
     * 
     * @function io
     * @param {Object} req - client request
     * @param {Integer} statusCode - response status code
     * @return {Object}
     */
    return function io(req, statusCode, end) {

      var diff = end[0] * 1e3 + end[1] * 1e-6;
      return {
        ip: req.remoteAddr || req.ip,
        method: req.method,
        status: statusCode,
        url: req.url,
        response: diff.toFixed(2)
      };
    };
  }
  /**
   * logger output after info builder
   * 
   * @function io
   * @param {Object} req - client request
   * @param {Integer} statusCode - response status code
   * @param {Array} end - high resolution time
   * @return {Object}
   */
  return function io(req, statusCode, end) {

    var diff = end[0] * 1e3 + end[1] * 1e-6;
    var out = {
      ip: req.remoteAddr || req.ip,
      method: req.method,
      status: statusCode,
      url: req.url,
      response: diff.toFixed(2)
    };
    for (var i = 0; i < l; ++i) {
      var p = promise[i];
      out[p[0]] = p[1](req);
    }
    return out;
  };
}

/**
 * function wrapper for multiple require
 * 
 * @function wrapper
 * @param {Function} log - logging function
 * @param {Object} my - parsed options
 * @param {Function} io - extra function
 * @return {Function}
 */
function wrapper(log, my, io) {

  var finished = require('on-finished');

  /**
   * end of job (closures). Get response time and status code
   * 
   * @function finale
   * @param {Object} req - client request
   * @param {Integer} statusCode - response status code
   * @param {Array} start - high resolution time
   * @return {Null}
   */
  function finale(req, statusCode, start) {

    req.start = start;
    return log(my.logger, io(req, statusCode, process.hrtime(start)));
  }

  if (my.deprecated) {
    /**
     * logging all route
     * 
     * @deprecated
     * @function deprecated
     * @param {Object} req - client request
     * @param {Object} res - response to client
     * @param {Function} next - next callback
     */
    return require('util').deprecate(function deprecated(req, res, next) {

      var start = process.hrtime();
      req.remoteAddr = req.ip;
      if (res._headerSent) { // function
        finale(req, res.statusCode, start); // after res.end()
      } else { // middleware
        var buffer = res.end;
        /**
         * middle of job. Set right end function (closure)
         * 
         * @function
         * @param {String} chunk - data sent
         * @param {String} encoding - data encoding
         * @return {Boolean}
         */
        res.end = function end(chunk, encoding) {

          res.end = buffer;
          var b = res.end(chunk, encoding);
          // res.end(chunk,encoding,finale) // callback available only
          // with node 0.11
          finale(req, res.statusCode, start); // write after sending
          // all stuff, instead of callback
          return b;
        };
      }

      return next ? next() : null;
    }, '`logger-request` option is deprecated');
  }

  if (my.functions) {
    /**
     * logging all route without next callback
     * 
     * @function logging
     * @param {Object} req - client request
     * @param {Object} res - response to client
     * @return {Null}
     */
    return function logging(req, res) {

      req.remoteAddr = req.ip;
      return finale(req, res.statusCode, process.hrtime());
    };
  }

  /**
   * logging all route
   * 
   * @function logging
   * @param {Object} req - client request
   * @param {Object} res - response to client
   * @param {Function} next - continue routes
   * @return {Null}
   */
  return function logging(req, res, next) {

    var start; // closure
    req.remoteAddr = req.ip;

    finished(res, function() {

      return finale(req, res.statusCode, start);
    });

    start = process.hrtime();
    return next();
  };
}

/**
 * option setting
 * 
 * @exports logger
 * @function logger
 * @param {Object} opt - various options. Check README.md
 * @return {Function|Object}
 */
function logger(opt) {

  var winston = require('winston');

  var options = opt || Object.create(null);
  var my = {
    filename: options.filename || false,
    daily: Boolean(options.daily),
    transports: Array.isArray(options.transports) ? options.transports : []
  };
  if (Boolean(options.deprecated)) {
    my.deprecated = true;
  } else if (Boolean(options.functions)) {
    my.functions = true;
  }

  // winston
  var optional = options.winston || Object.create(null);
  // default option for File transport and Console
  optional.logger = String(optional.logger || 'logger-request');
  optional.level = String(optional.level || 'info');
  optional.timestamp = optional.timestamp || true;
  optional.maxsize = Number(optional.maxsize) || 8388608;
  optional.maxFiles = Number(optional.maxFiles) || null;
  optional.json = optional.json === false ? false : true;
  optional.raw = optional.raw === false ? false : true;
  my.logger = optional.logger;

  var log = new winston.Logger(); // without transport

  if (my.filename) {
    optional.filename = require('path').resolve(String(my.filename));
    if (my.daily) {
      optional.prepend = true;
      log.add(require('winston-daily-rotate-file'), optional);
    } else {
      log.add(winston.transports.File, optional);
    }
  }

  if (Boolean(options.console)) {
    log.add(winston.transports.Console, optional);
  }

  for (var i = 0, ii = my.transports.length; i < ii; ++i) {
    log.add(my.transports[i], optional);
  }
  log = log[optional.level]; // extract logger level function

  if (Boolean(options.standalone)) {
    return log;
  }

  // custom
  options = options.custom || Object.create(null);

  return wrapper(log, my, __info({
    pid: Boolean(options.pid),
    bytesReq: Boolean(options.bytesReq),
    bytesRes: Boolean(options.bytesRes),
    referer: Boolean(options.referer),
    auth: Boolean(options.auth),
    transfer: Boolean(options.transfer),
    agent: Boolean(options.agent),
    lang: Boolean(options.lang),
    cookie: Boolean(options.cookie),
    headers: Boolean(options.headers),
    version: Boolean(options.version),
    callback: typeof options.callback == 'function' ? options.callback : false
  }));
}
module.exports = logger;
