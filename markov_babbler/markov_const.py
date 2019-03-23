# -*- coding: UTF-8 -*-

BEGIN_SYMBOL = "$$$BEGIN$$$"
END_SYMBOL = "$$$END$$$"


### Give the bot a little bit of decorum against the worst of the internet
BLACKLIST = [
    "nigger", "spic", "coon", "kike", "cunt", "twat", "faggot", "nigga", "niggas", u"…", "hitler",
]

PURGE_LIST = [
    "RT"
]


CLEANUP_REGEXES = [(r"[ ]([-])[ ]", "'"),
                    (r"[ ]([%;:,.?!)])", r"\1"),
                   (r"([(])[ ]", r"\1")]