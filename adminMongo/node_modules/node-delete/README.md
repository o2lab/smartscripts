# node-delete

[![NPM](https://nodei.co/npm/node-delete.png?downloads=true&downloadRank=true&stars=true)](https://nodei.co/npm/node-delete/)

Delete files, folders in Nodejs using globs.

It also protects you against deleting the current working directory and above.


# Install

```sh
$ npm install --save node-delete
```

# Usage

```js
var del = require('node-delete');

del(['tmp/*.js', '!tmp/d.js'], function (err, paths) {
	console.log('Deleted files/folders:\n', paths.join('\n'));
});
```

# Beware

The glob pattern `**` matches all children and *the parent*.

So this won't work:

```js
del.sync(['public/assets/**', '!public/assets/goat.png']);
```

You have to explicitly ignore the parent directories too:

```js
del.sync(['public/assets/**', '!public/assets', '!public/assets/goat.png']);
```


# API

## del(patterns, [options], callback)
## del.sync(patterns, [options])

The async method gets an array of deleted paths as the second argument in the callback, while the sync method returns the array.

### options

Type: `object`

See the node-glob [options](https://github.com/isaacs/node-glob#options).

### options.force

Type: `boolean`  
Default: `false`

Allow deleting the current working directory and files/folders outside it.

# Github

See https://github.com/duyetdev/node-delete

# License
MIT License

Copyright (c) 2015 Van-Duyet Le

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

[![Bitdeli Badge](https://d2weczhvl823v0.cloudfront.net/duyetdev/node-delete/trend.png)](https://bitdeli.com/free "Bitdeli Badge")

