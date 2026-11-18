(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* Writes the dune rules of this directory (see dune and README.md). The tree
   has them, in run/dune.inc, and each alias of the directory runs this program
   and compares what it writes with that file. No build of the compiler runs the
   program, nor make runtest, so that where it stops one of the two aliases
   fails and no build of the compiler (README.md, "What a rule does", says what
   dune's own aliases all, default and check do).

   gen_rules.exe BUILD_RULES LONG NOT_RUN DRIVER_STATUS X.own_expected ...

   BUILD_RULES is the text of the rules by which the build of the tree compiles
   the standard library, as stdlib/dune_rules/gen.exe writes it. For each unit
   that those rules compile to native code, the rules written here give
   stdlib_run.sh the real source, the files that the build's rule depends on and
   the words of the build's command, and compare what the driver says of the
   unit with the stored file. Nothing of the build's command is written in this
   program. Every stanza of the build's rules must be of a kind that this
   program names ([read_rule]), those that it has no use for among them: a
   stanza that is not, or a rule that compiles a unit and has not the shape that
   stdlib_run.sh is written for, stops it, with the stanza printed. *)

(* The shapes that this program knows are matched, and anything else is refused:
   the matches are fragile on purpose. *)
[@@@ocaml.warning "-4"]

type sexp =
  | Atom of string
  | Quoted of string
  | List of sexp list

exception Bad of string

let bad fmt = Printf.ksprintf (fun s -> raise (Bad s)) fmt

(* A reader of the text of dune rules: lists, quoted strings with the escapes
   that the generator of the build writes, and atoms. *)
let parse text =
  let n = String.length text in
  let pos = ref 0 in
  let rec skip () =
    if !pos < n
    then
      match text.[!pos] with
      | ' ' | '\t' | '\n' | '\r' ->
        incr pos;
        skip ()
      | ';' ->
        while !pos < n && text.[!pos] <> '\n' do
          incr pos
        done;
        skip ()
      | _ -> ()
  in
  let quoted () =
    let b = Buffer.create 64 in
    let rec go () =
      if !pos >= n then bad "the text ends inside a string";
      match text.[!pos] with
      | '"' -> incr pos
      | '\\' ->
        if !pos + 1 >= n then bad "the text ends inside a string";
        (match text.[!pos + 1] with
        | 'n' -> Buffer.add_char b '\n'
        | '"' -> Buffer.add_char b '"'
        | '\\' -> Buffer.add_char b '\\'
        | c -> bad "an escape in a string that this reader has not: \\%c" c);
        pos := !pos + 2;
        go ()
      | c ->
        Buffer.add_char b c;
        incr pos;
        go ()
    in
    go ();
    Buffer.contents b
  in
  let atom () =
    let start = !pos in
    let in_atom c =
      match c with
      | ' ' | '\t' | '\n' | '\r' | '(' | ')' | '"' | ';' -> false
      | _ -> true
    in
    while !pos < n && in_atom text.[!pos] do
      incr pos
    done;
    String.sub text start (!pos - start)
  in
  let rec one () =
    skip ();
    if !pos >= n then bad "the text ends inside a list";
    match text.[!pos] with
    | '(' ->
      incr pos;
      List (many ())
    | ')' -> bad "a closing parenthesis with no list open"
    | '"' ->
      incr pos;
      Quoted (quoted ())
    | _ -> Atom (atom ())
  and many () =
    skip ();
    if !pos >= n then bad "the text ends inside a list";
    if text.[!pos] = ')'
    then (
      incr pos;
      [])
    else
      let first = one () in
      first :: many ()
  in
  let rec top acc =
    skip ();
    if !pos >= n then List.rev acc else top (one () :: acc)
  in
  top []

let quote s =
  let b = Buffer.create (String.length s + 2) in
  Buffer.add_char b '"';
  String.iter
    (fun c ->
      match c with
      | '"' -> Buffer.add_string b "\\\""
      | '\\' -> Buffer.add_string b "\\\\"
      | '\n' -> Buffer.add_string b "\\n"
      | c -> Buffer.add_char b c)
    s;
  Buffer.add_char b '"';
  Buffer.contents b

let rec show sexp =
  match sexp with
  | Atom a -> a
  | Quoted q -> quote q
  | List l -> "(" ^ String.concat " " (List.map show l) ^ ")"

let ends_with ~suffix s =
  let n = String.length s and m = String.length suffix in
  n >= m && String.sub s (n - m) m = suffix

let without ~suffix s = String.sub s 0 (String.length s - String.length suffix)

(* What the rules of the build say. *)

(* How the build makes each source in its directory: after a line that gives its
   name and the number 2, or as it is. *)
type form =
  | Numbered
  | Copied

let forms : (string, form) Hashtbl.t = Hashtbl.create 200

(* The interface of each unit: stdlib__List gives list.mli. *)
let interfaces : (string, string) Hashtbl.t = Hashtbl.create 100

type compiled =
  { unit_name : string;
    source : string;
    files : string list;
    words : sexp list
  }

let compiled : compiled list ref = ref []

let boot = "%{exe:../../boot_ocamlopt.exe}"

let stdlib_flags = "%{read-lines:../../ocamlopt_stdlib_flags.txt}"

let awk_in_the_build = "awk -f %{dep:../expand_module_aliases.awk}"

let awk_here = "awk -f ../expand_module_aliases.awk"

let field name rule fields =
  match
    List.filter_map
      (fun f ->
        match f with
        | List (Atom head :: rest) when head = name -> Some rest
        | _ -> None)
      fields
  with
  | [rest] -> rest
  | _ -> bad "a rule that has not one field %s: %s" name (show rule)

(* The rule that compiles a unit to native code: it must have the shape that
   stdlib_run.sh is written for. *)
let compilation rule fields unit_name =
  let wrong what = bad "%s, in the rule of %s: %s" what unit_name (show rule) in
  List.iter
    (fun f ->
      match f with
      | List (Atom ("targets" | "deps" | "action") :: _) -> ()
      | _ -> wrong "a field that is not targets, deps or action")
    fields;
  let source, files =
    match field "deps" rule fields with
    | List [Atom ":src"; Atom source] :: files ->
      if not (ends_with ~suffix:".ml" source)
      then wrong "the source is not an .ml file";
      let files =
        List.map
          (fun f ->
            match f with
            | Atom f
              when (ends_with ~suffix:".cmi" f || ends_with ~suffix:".cmx" f)
                   && f <> unit_name ^ ".cmx"
                   && not (String.contains f '/') ->
              f
            | _ ->
              wrong "it depends on a file that is no .cmi or .cmx of a unit")
          files
      in
      if not (List.mem (unit_name ^ ".cmi") files)
      then wrong "it does not depend on the .cmi of the unit";
      source, files
    | _ -> wrong "the first of what it depends on is not (:src X.ml)"
  in
  let words =
    match field "action" rule fields with
    | [List (Atom "run" :: Atom program :: words)] ->
      if program <> boot then wrong "the program is not the boot compiler";
      words
    | _ -> wrong "the action is not one run of a program"
  in
  let count w = List.length (List.filter (fun x -> x = w) words) in
  if count (Atom stdlib_flags) <> 1
  then wrong "the flags of ocamlopt_stdlib_flags.txt are not read once";
  if count (Atom "-cmi-file") <> 1 then wrong "-cmi-file is not given once";
  let rec check words =
    match words with
    | [Atom "-o"; Atom target; Atom "-c"; Atom "%{src}"] ->
      if target <> unit_name ^ ".cmx"
      then wrong "-o does not name the .cmx of the unit"
    | Atom "-cmi-file" :: Atom cmi :: rest ->
      if cmi <> unit_name ^ ".cmi"
      then wrong "-cmi-file does not name the .cmi of the unit";
      check rest
    | Atom "-pp" :: Quoted command :: rest ->
      if command <> awk_in_the_build
      then wrong "-pp is given a command that this program does not know";
      check rest
    | Atom word :: rest ->
      if word = "-pp" then wrong "-pp is not followed by a quoted command";
      if word = "-o" || word = "-c"
      then wrong "the command does not end with -o UNIT.cmx -c %{src}";
      if
        word <> stdlib_flags
        && String.length word >= 2
        && String.contains word '{'
      then wrong ("a word that this program does not know, " ^ word);
      check rest
    | (Quoted _ | List _) :: _ ->
      wrong "a word of the command that is quoted, or a list"
    | [] -> wrong "the command does not end with -o UNIT.cmx -c %{src}"
  in
  check words;
  compiled := { unit_name; source; files; words } :: !compiled

(* The name of a file after its last dot. *)
let extension name =
  match String.rindex_opt name '.' with
  | None -> ""
  | Some i -> String.sub name (i + 1) (String.length name - i - 1)

let all_have_none extensions targets =
  List.for_all (fun t -> not (List.mem (extension t) extensions)) targets

(* The files that the build copies into its directory and that are of no unit:
   they are named here one by one. *)
let copied_and_not_read =
  [ "../../runtime/lib{asm,caml}run*.{a,so}";
    "../../Makefile.config";
    "../dune_rules/ld.conf" ]

(* What a rule makes: the names in its one field targets, or target. *)
let targets_of rule fields =
  match
    List.filter_map
      (fun f ->
        match f with
        | List (Atom ("targets" | "target") :: targets) -> Some targets
        | _ -> None)
      fields
  with
  | [targets] ->
    List.map
      (fun t ->
        match t with
        | Atom t -> t
        | Quoted _ | List _ ->
          bad "a rule with a target that is not a plain name: %s" (show rule))
      targets
  | _ -> bad "a rule that has not one field targets or target: %s" (show rule)

(* Each stanza of the build's rules is read here, and EVERY ONE MUST BE OF A
   KIND THAT THIS FUNCTION NAMES: the two ways in which the build puts a source
   in its directory; the rule that compiles a unit to native code, which
   [compilation] reads whole; the rule that makes the .cmi of a unit; and, of no
   use here and let pass by name, the rules of the bytecode, of the three
   archives and of runtime-launch-info, three files that are copied, and the
   alias stdlib, which names files and makes none. Any other stanza stops the
   program, and so does a rule that makes a .cmx or a .o in another way than
   (targets UNIT.o UNIT.cmx): a unit that the build compiles cannot be left out
   of the rules written here because its rule has a shape that is not known. *)
let read_rule rule =
  match rule with
  | List [Atom "rule"; List [Atom "with-stdout-to"; Atom made; action]]
    when ends_with ~suffix:".ml" made || ends_with ~suffix:".mli" made -> (
    let line = "# 2 \"" ^ made ^ "\"\n" in
    match action with
    | List
        [ Atom "progn";
          List [Atom "echo"; Quoted said];
          List [Atom "cat"; Atom from] ]
      when said = line && from = "../" ^ made ->
      Hashtbl.replace forms made Numbered
    | _ ->
      bad "a source that the build makes in a way not known here: %s"
        (show rule))
  | List [Atom "copy_files"; Atom from]
    when ends_with ~suffix:".ml" from || ends_with ~suffix:".mli" from ->
    let made = Filename.basename from in
    if from <> "../" ^ made || String.contains made '*'
    then
      bad "a source that the build copies in a way not known here: %s"
        (show rule);
    Hashtbl.replace forms made Copied
  | List [Atom "copy_files"; Atom from] when List.mem from copied_and_not_read
    ->
    ()
  | List
      [Atom "alias"; List [Atom "name"; Atom "stdlib"]; List (Atom "deps" :: _)]
    ->
    ()
  | List (Atom "rule" :: fields) -> (
    let targets = targets_of rule fields in
    let all_have extensions =
      List.for_all (fun t -> List.mem (extension t) extensions) targets
    in
    match targets with
    | [o; cmx]
      when ends_with ~suffix:".cmx" cmx && o = without ~suffix:".cmx" cmx ^ ".o"
      ->
      compilation rule fields (without ~suffix:".cmx" cmx)
    | _ when not (all_have_none ["cmx"; "o"] targets) ->
      bad
        "a rule that makes a .cmx or a .o and has not the targets UNIT.o \
         UNIT.cmx: %s"
        (show rule)
    | cmi :: _
      when ends_with ~suffix:".cmi" cmi && all_have ["cmi"; "cmsi"; "cmti"] -> (
      match field "deps" rule fields with
      | List [Atom ":src"; Atom source] :: _
        when ends_with ~suffix:".mli" source ->
        Hashtbl.replace interfaces (without ~suffix:".cmi" cmi) source
      | _ ->
        bad "a rule that makes a .cmi and has not (:src X.mli): %s" (show rule))
    | _ :: _ when all_have ["cmo"; "cms"; "cmt"] -> ()
    | _ :: _ when all_have ["cmxa"; "a"; "cma"] -> ()
    | ["runtime-launch-info"] -> ()
    | _ ->
      bad "a rule of the build whose targets this program does not know: %s"
        (show rule))
  | _ ->
    bad "a stanza of the build's rules that this program does not know: %s"
      (show rule)

(* The lists of targets, anywhere in a stanza, that name a .cmx. *)
let rec makers_of_cmx stanza =
  match stanza with
  | Atom _ | Quoted _ -> 0
  | List (Atom ("targets" | "target") :: targets) ->
    if
      List.exists
        (fun t ->
          match t with
          | Atom t -> ends_with ~suffix:".cmx" t
          | Quoted _ | List _ -> false)
        targets
    then 1
    else 0
  | List parts -> List.fold_left (fun n part -> n + makers_of_cmx part) 0 parts

(* The lists of this directory: one name a line, and whatever follows it on the
   line; a line that begins with # is a comment. *)
let read_file path =
  let channel = open_in_bin path in
  let text = really_input_string channel (in_channel_length channel) in
  close_in channel;
  text

let read_list path =
  let entries =
    List.filter_map
      (fun line ->
        let line = String.trim line in
        if line = "" || line.[0] = '#'
        then None
        else
          match String.index_opt line ' ' with
          | None -> Some (line, "")
          | Some i ->
            Some
              ( String.sub line 0 i,
                String.trim
                  (String.sub line (i + 1) (String.length line - i - 1)) ))
      (String.split_on_char '\n' (read_file path))
  in
  let names = List.map fst entries in
  List.iter
    (fun name ->
      if List.length (List.filter (fun other -> other = name) names) > 1
      then bad "%s names %s twice" path name)
    names;
  entries

(* The rules. They are for run/, so they name the files of this directory with
   ../ and those of the tree from its root. *)
let root = "../../../../../../"

(* The head of the file of rules. *)
let head =
  {|;; The rules of ../ for each unit of the standard library that the build of
;; the tree compiles to native code (../README.md). ../gen_rules.exe wrote
;; this file from the rules of the build. Each of the two aliases of the
;; directory compares it with what that program writes now, and `dune
;; promote` then takes the new rules: it is not to be changed by hand.
|}

let rules_of ~long ~status c =
  let name = without ~suffix:".ml" c.source in
  let form =
    match Hashtbl.find_opt forms c.source with
    | Some Numbered -> "numbered"
    | Some Copied -> "copied"
    | None -> bad "no rule of the build makes %s in its directory" c.source
  in
  let source_of file =
    if ends_with ~suffix:".cmi" file
    then
      match Hashtbl.find_opt interfaces (without ~suffix:".cmi" file) with
      | Some interface -> interface
      | None -> bad "no rule of the build makes %s from an interface" file
    else
      let unit_name = without ~suffix:".cmx" file in
      match List.find_opt (fun c -> c.unit_name = unit_name) !compiled with
      | Some c -> c.source
      | None -> bad "no rule of the build makes %s" file
  in
  let sources =
    List.sort_uniq compare (List.map source_of c.files)
    |> List.map (fun source -> "  " ^ root ^ "stdlib/" ^ source)
  in
  let awk = ref "-" in
  let words =
    List.map
      (fun word ->
        match word with
        | Atom w when w = stdlib_flags ->
          "%{read-lines:" ^ root ^ "ocamlopt_stdlib_flags.txt}"
        | Atom "%{src}" -> "./" ^ c.source
        | Atom w -> w
        | Quoted _ ->
          awk := "%{dep:" ^ root ^ "stdlib/expand_module_aliases.awk}";
          quote awk_here
        | List _ -> assert false)
      c.words
  in
  let lines words =
    (* the words of a command, a few to a line *)
    let rec go line acc words =
      match words with
      | [] -> List.rev (if line = "" then acc else line :: acc)
      | word :: rest ->
        if line = ""
        then go word acc rest
        else if String.length line + 1 + String.length word > 70
        then go word (line :: acc) rest
        else go (line ^ " " ^ word) acc rest
    in
    String.concat "\n" (List.map (fun line -> "    " ^ line) (go "" [] words))
  in
  let alias = if long then "runtest-stdlib-long" else "runtest-stdlib" in
  Printf.printf
    {|
(rule
 (enabled_if
  (= %%{context_name} "main"))
 (deps
  %%{ocamlopt}
  (glob_files %%{ocaml_where}/*.cmi)
  (glob_files %%{ocaml_where}/*.cmx)
  (glob_files %%{ocaml_where}/*.o)
  (glob_files %%{ocaml_where}/*.ml)
%s)
 (action
  (with-stdout-to
   %s.own_output
   (run sh %%{dep:../stdlib_run.sh} %%{ocamlopt}
    %%{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %%{dep:%stools/objinfo.exe} %%{ocaml_where}
    %%{dep:%sstdlib/%s}
    %s %s %s
    %s
%s
    --
%s))))

(rule
 (alias %s)
 (enabled_if
  (= %%{context_name} "main"))
 (action
  (diff ../%s.own_expected
   %s.own_output)))
|}
    (String.concat "\n" sources)
    name root root c.source form c.unit_name status !awk (lines c.files)
    (lines words) alias name name

let () =
  match Array.to_list Sys.argv with
  | _ :: build_rules :: long :: not_run :: driver_status :: stored -> (
    try
      let stanzas = parse (read_file build_rules) in
      List.iter read_rule stanzas;
      (* Beside the reading: every list of targets, anywhere in the text, that
         names a .cmx is counted, and each must be of a rule that was read as
         the compilation of a unit. *)
      let makers = List.fold_left (fun n s -> n + makers_of_cmx s) 0 stanzas in
      if makers <> List.length !compiled
      then
        bad
          "%d lists of targets in the rules of the build name a .cmx, and %d \
           rules were read as the compilation of a unit"
          makers (List.length !compiled);
      let compiled =
        List.sort (fun a b -> compare a.source b.source) !compiled
      in
      if compiled = [] then bad "the rules of the build compile no unit";
      let names = List.map (fun c -> without ~suffix:".ml" c.source) compiled in
      let long = read_list long in
      let not_run = read_list not_run in
      let driver_status = read_list driver_status in
      let known list_name entries =
        List.iter
          (fun (name, _) ->
            if not (List.mem name names)
            then bad "%s names %s, no unit of the library" list_name name)
          entries
      in
      known "long.txt" long;
      known "not_run.txt" not_run;
      known "driver_status.txt" driver_status;
      let in_list entries name = List.mem_assoc name entries in
      List.iter
        (fun name ->
          let lists =
            List.filter
              (fun entries -> in_list entries name)
              [long; not_run; driver_status]
          in
          if List.length lists > 1
          then bad "%s is in two of the three lists of this directory" name)
        names;
      let stored =
        List.map
          (fun path ->
            let base = Filename.basename path in
            if not (ends_with ~suffix:".own_expected" base)
            then bad "%s is not a stored file X.own_expected" path;
            without ~suffix:".own_expected" base)
          stored
      in
      List.iter
        (fun name ->
          if not (List.mem name names)
          then bad "%s.own_expected is of no unit that the build compiles" name;
          if in_list not_run name
          then bad "%s.own_expected is of a unit that not_run.txt names" name)
        stored;
      print_string head;
      List.iter
        (fun c ->
          let name = without ~suffix:".ml" c.source in
          if in_list not_run name
          then ()
          else (
            if not (List.mem name stored)
            then
              bad "the build compiles %s, and there is no %s.own_expected here"
                c.source name;
            let status =
              match List.assoc_opt name driver_status with
              | None -> "0"
              | Some status -> (
                match int_of_string_opt status with
                | Some n when n > 0 && n < 256 -> string_of_int n
                | _ ->
                  bad "driver_status.txt gives %s no status from 1 to 255" name)
            in
            rules_of ~long:(in_list long name) ~status c))
        compiled
    with Bad why ->
      prerr_endline ("gen_rules: " ^ why);
      exit 1)
  | _ ->
    prerr_endline
      "usage: gen_rules.exe BUILD_RULES LONG NOT_RUN DRIVER_STATUS \
       X.own_expected ...";
    exit 2
