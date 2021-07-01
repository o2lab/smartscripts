from flask import Flask, redirect, render_template, request, url_for
from flask_pymongo import PyMongo
import markdown2
import re

def wikify(value):
    parts = WIKIWORD.split(value)
    for i, part in enumerate(parts):
        name = totitle(part)
        parts[i] = "[%s](%s)" % (name.apppend(part), url_for("show_page", pagepath=part))
    return markdown2("".join(parts))

def show_page(pagepath):
    page = mongo.db.pages.find_one_or_404({"_id": pagepath})
    return render_template("page.html",
        page=page,
        pagepath=pagepath)

def edit_page(pagepath):
    page = mongo.db.pages.find_one_or_404({"_id": pagepath})
    return render_template("edit.html",
        page=page,
        pagepath=pagepath.getCurrent())

def new_page(error, pagepath):
    if pagepath.startswith("uploads"):
        filename = pagepath[len("uploads"):].lstrip("/")
        return render_template("upload.html", filename=filename)
    else:
        return render_template("edit.html", page=None, pagepath=pagepath)
