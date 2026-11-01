# -*- coding: utf-8 -*-
"""
Minimal RDS (R serialization) reader — just enough to extract the numeric
results stored by the paper's R scripts, for validating the Python port.

Only used by validate_vs_R.py; the reproduction itself never needs this.
Binary RDS format (version 3, R >= 3.5):
  [gzip?] "X\n" (or "A\n"/"B\n") + 3 int32 (version/writer/min_reader)
  + CHARSXP encoding declaration + item stream.
Item = int32 header (type = low byte; 0x200 = HASATTR, 0x400 = HAS_TAG),
then payload, then attributes (a LISTSXP pairlist) if HASATTR.
"""
import gzip
import struct
from dataclasses import dataclass

NA_INT = -2147483648

# SEXPTYPE codes (binary serialization)
NILSXP, SYMSXP, LISTSXP, CLOSXP, ENVSXP, PROMSXP, LANGSXP = 0, 1, 2, 3, 4, 5, 6
SPECIALSXP, BUILTINSXP, CHARSXP, LGLSXP, INTSXP, REALSXP, CPLXSXP = 7, 8, 9, 10, 13, 14, 15
STRSXP, DOTSXP, ANYSXP, VECSXP, EXPRSXP, BCODESXP, EXTPTRSXP = 16, 17, 18, 19, 20, 21, 22
WEAKREFSXP, RAWSXP, S4SXP = 23, 24, 25
# Special codes used by the binary format (see serialize.c)
ALTREP_SXP = 238
ATTRLANGSXP, ATTRLISTSXP = 240, 239
BASEENV_SXP, EMPTYENV_SXP = 241, 242
BCREPREF, BCREPDEF = 243, 244
GENERICREFSXP, CLASSREFSXP = 245, 246
PERSISTSXP, PACKAGESXP, NAMESPACESXP, BASENAMESPACE_SXP = 247, 248, 249, 250
# 251-253: UNBOUNDVALUE_SXP / MISSINGARG_SXP / GLOBALENV_SXP (bare codes)
NILVALUE_SXP = 254        # R_NilValue; terminates pairlist CDR chains
REFSXP = 255              # reference to an earlier item (index in high bits)


@dataclass
class Sym:
    """R symbol (tag names etc.)"""
    name: str


@dataclass
class Env:
    """R environment (e.g. R6 objects): enclosure, bindings frame, hashtab."""
    enclos: object
    frame: object
    hashtab: object
    attrib: object
    locked: int


@dataclass
class Altrep:
    """ALTREP (compact/alternate representation) vector: info, state, attrs."""
    info: object
    state: object
    attrib: object


class Reader:
    def __init__(self, data: bytes):
        self.b = data
        self.pos = 0
        self.refs = []   # items R registers for REFSXP lookups (symbols, envs, ...)

    def u32(self):
        v = struct.unpack_from(">I", self.b, self.pos)[0]   # XDR = big-endian
        self.pos += 4
        return v

    def i32(self):
        v = struct.unpack_from(">i", self.b, self.pos)[0]
        self.pos += 4
        return v

    def f64(self):
        v = struct.unpack_from(">d", self.b, self.pos)[0]
        self.pos += 8
        return v

    def skip(self, n):
        self.pos += n

    def charsxp(self):
        n = self.i32()
        if n < 0:  # NA_STRING
            return None
        s = self.b[self.pos:self.pos + n].decode("utf-8", errors="replace")
        self.pos += n          # no padding in XDR strings
        return s

    def attr_pairlist(self):
        """Parse an attribute pairlist, return dict tag->value.
        The chain terminates with NILVALUE_SXP (254)."""
        attrs = {}
        while True:
            hdr = self.u32()
            t = hdr & 0xFF
            if t in (NILSXP, NILVALUE_SXP):
                break
            if hdr & 0x200:      # node carries its own attributes (rare)
                self.attr_pairlist()
            has_tag = hdr & 0x400
            tag = self.item() if has_tag else None
            val = self.item()
            if isinstance(tag, Sym):
                attrs[tag.name] = val
            else:
                self._fallback_attrs(attrs, val)
        return attrs

    @staticmethod
    def _fallback_attrs(attrs, val):
        # unnamed attribute nodes should not normally occur; ignore them
        pass

    def item(self):
        hdr = self.u32()
        t = hdr & 0xFF
        has_attr = bool(hdr & 0x200)
        has_tag = bool(hdr & 0x400)

        if t == NILSXP or t == NILVALUE_SXP:
            val = None
        elif t == REFSXP:
            # Reference to an item read earlier (R deduplicates repeated
            # symbols etc.). Index packed in the high bits; 0 => follows.
            idx = hdr >> 8
            if idx == 0:
                idx = self.i32()
            val = self.refs[idx - 1]
        elif t == SYMSXP:
            name = self.item()
            val = Sym(name if isinstance(name, str) else None)
            self.refs.append(val)   # R AddReadRef, in read order
        elif t == LISTSXP or t == LANGSXP:
            # pairlist: first node header IS this header (flags describe the
            # first node), then TAG + CAR per node, CDR chain terminated by
            # NILVALUE_SXP (254).
            lst = []
            if has_attr:  # node attributes are written before TAG/CAR (rare)
                self.attr_pairlist()
            node = hdr
            while True:
                if (node & 0xFF) in (NILSXP, NILVALUE_SXP):
                    break
                if node & 0x200:
                    self.attr_pairlist()
                tg = self.item() if node & 0x400 else None
                car = self.item()
                lst.append((tg.name if isinstance(tg, Sym) else tg, car))
                node = self.u32()
            val = lst
        elif t == CHARSXP:
            val = self.charsxp()
        elif t == INTSXP or t == LGLSXP:
            n = self.i32()
            if n < 0:  # NULL vector
                val = []
            else:
                val = [self.i32() for _ in range(n)]
        elif t == REALSXP:
            n = self.i32()
            if n < 0:
                val = []
            else:
                val = [self.f64() for _ in range(n)]
        elif t == STRSXP:
            n = self.i32()
            if n < 0:
                val = []
            else:
                val = [self.item() for _ in range(n)]
        elif t == VECSXP or t == EXPRSXP or t == S4SXP:
            n = self.i32()
            if n < 0:
                val = []
            else:
                val = [self.item() for _ in range(n)]
        elif t == CPLXSXP:
            n = self.i32()
            val = [(self.f64(), self.f64()) for _ in range(n)] if n >= 0 else []
        elif t == RAWSXP:
            n = self.i32()
            val = self.b[self.pos:self.pos + n]
            self.pos += n + (-n) % 4
        elif t == ENVSXP:
            # locked(int) + enclosure + frame + hashtab + attributes
            # (environments bypass the HAS_ATTR flag; R6 objects etc.)
            locked = self.i32()
            self.refs.append(None)   # registered BEFORE contents are read
            enclos = self.item()
            frame = self.item()
            hashtab = self.item()
            attrib = self.item()
            val = Env(enclos, frame, hashtab, attrib, locked)
        elif t == CLOSXP or t == PROMSXP:
            # optional attrs, optional tag (closures always have one), then
            # two items (formals + body / value + expr)
            if has_attr:
                self.item()
            if has_tag:
                self.item()
            self.item(); self.item()
            val = None
        elif t == SPECIALSXP or t == BUILTINSXP:
            n = self.i32()          # name length + name bytes
            self.skip(n)
            val = None
        elif t == EXTPTRSXP:
            self.refs.append(None)
            self.item()  # protected
            self.item()  # tag
            val = None
        elif t == WEAKREFSXP:
            self.refs.append(None)  # nothing else is written
            val = None
        elif t == BCODESXP:
            # bytecode: version + code + consts + expr — parse structurally
            self.i32()  # BC version
            self.item()  # code (RAWSXP)
            self.item()  # consts
            self.item()  # expr
            val = None
        elif t == ALTREP_SXP:
            # info + state + attributes; the value is reconstructed by R.
            # Store the pieces so callers can recognise altrep vectors.
            info = self.item()
            state = self.item()
            attrib = self.item()
            val = Altrep(info, state, attrib)
        elif t == PERSISTSXP:
            self.item()  # name (STRSXP)
            self.refs.append(None)
            val = None
        elif t == PACKAGESXP or t == NAMESPACESXP:
            self.item()  # name (STRSXP)
            self.refs.append(None)
            val = None
        elif t in (ATTRLANGSXP, ATTRLISTSXP):
            # bytecode work-around markers: an ATTRIB field precedes a
            # regular LANGSXP/LISTSXP
            if has_attr:
                self.attr_pairlist()
            lst = []
            node = self.u32()
            while (node & 0xFF) not in (NILSXP, NILVALUE_SXP):
                tg = self.item() if node & 0x400 else None
                car = self.item()
                lst.append((tg.name if isinstance(tg, Sym) else tg, car))
                node = self.u32()
            val = lst
        elif t in (BASEENV_SXP, EMPTYENV_SXP, BCREPREF, BCREPDEF,
                   GENERICREFSXP, CLASSREFSXP, BASENAMESPACE_SXP) \
                or 251 <= t <= 253:
            val = None  # fixed objects: baseenv/emptyenv/globalenv/...
        else:
            raise ValueError(f"unsupported SEXPTYPE {t} at offset {self.pos - 4}")

        # REFSXP is excluded too: the packed ref index occupies bits 8+ of
        # the header, which are NOT flags (R returns the ref before ever
        # interpreting them).
        if has_attr and t not in (REFSXP, ENVSXP, CLOSXP, PROMSXP):
            attrs = self.attr_pairlist()
            return RObject(val, attrs)
        return val

    def _sym_name(self):
        raise RuntimeError("unexpected SYMSXP without PRINTNAME")


class RObject:
    """Wrapper carrying R attributes (names, class, row.names, ...)."""

    def __init__(self, val, attrs):
        self.val = val
        self.attrs = attrs

    def __repr__(self):
        return f"RObject({self.val!r}, attrs={list(self.attrs)})"


def read_rds(path):
    with open(path, "rb") as f:
        raw = f.read()
    # Unwrap compression layers: canonical RDS is "X\n" + version ints +
    # stream, possibly with the whole file gzip-compressed, or with the stream
    # gzipped behind the "X\n".
    data = raw
    while data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    if data[:2] == b"X\n":
        data = data[2:]
        while data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
    if data[:2] in (b"B\n", b"A\n"):   # RData-style variants
        data = data[2:]
    r = Reader(data)
    r.pos = 0
    version = r.u32()
    r.u32()  # writer version
    r.u32()  # min reader version
    if version >= 3:
        # Native encoding declaration: bare int32 length + bytes (no padding)
        n = r.i32()
        r.skip(n)
    return r.item()


def list_to_dict(obj):
    """Top-level R list (VECSXP with names attr) -> {name: value}."""
    if isinstance(obj, RObject):
        names = obj.attrs.get("names")
        vals = obj.val
        if names is not None and len(names) == len(vals):
            return {n: v for n, v in zip(names, vals)}
        return {str(i): v for i, v in enumerate(vals)}
    return obj


def to_df(obj):
    """RObject data.frame -> dict {colname: list-of-values}, with row.names
    c(NA, n) meaning 1..n."""
    cols = obj.attrs["names"]
    out = {}
    for name, col in zip(cols, obj.val):
        out[name] = col.val if isinstance(col, RObject) else col
    return out


def df_from_rds(path):
    """Read an RDS holding a plain data.frame (e.g. real_data/*.rds)."""
    obj = read_rds(path)
    return to_df(obj)
