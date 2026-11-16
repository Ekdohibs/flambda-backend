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

(* The compiler, with the check of the test tool (../flsemantics.ml) made of the
   two terms of its own run. It is the driver of the compiler
   (driver/oxcaml_main.ml) with two hooks registered (driver/compiler_hooks.mli)
   and any closed standard descriptor opened first: it takes the compiler's
   command line, and is meant to compile as the compiler does: see below.

   The first mode of the tool (../flsemantics_main.ml) checks a pair that the
   tool makes: it reads the text that [-drawfexpr] prints of the term given to
   Simplify, and runs Simplify on what the reader made of the text. The text has
   not got all that the term has, and Simplify reads some of what is lost
   (../../semantics/ocaml/README.md, "What a validation does not say"). Here
   there is no text, no reader and no second Simplify.

   The hook [Raw_flambda2] is given the term that the compiler gives to Simplify
   (middle_end/flambda2/flambda2.ml, [flambda_to_flambda0], before anything else
   is done with the term). The driver keeps the term, and converts and checks
   nothing until Simplify has returned. In particular it does not ask there
   whether the term is well-formed, as the tool does of the term that it has
   read: that question converts the term, the conversion opens its binders, and
   opening a binder makes names. The numbers of the names that Simplify then
   makes would be later ones than in a compilation by the compiler, and what
   Simplify makes can depend on those numbers: the driver would check a
   compilation that is not the compiler's. (It did, until this was mended: it
   asked the question in this hook.)

   The hook [Flambda2] is given the term that the compiler's own Simplify
   returned (the same function, after [Simplify.run]). There the driver makes a
   child process ([Unix.fork]), and the check is made in the child: it asks
   whether the term that was kept is well-formed
   ([Flsemantics.well_formed_before]), and, if that answers yes, as it does also
   of a term that the conversion refuses, gives the two terms, as they are in
   memory, to [Flsemantics.check]: both are converted to the model, both are
   run, what the runs come to is compared, and the validator is asked. Nothing
   is done to either term before that: what the tool does to a term that it has
   read from a text ([Flsemantics.as_read_from_text]) is for what the text has
   lost, and is not done here. The same questions are asked in the same order,
   but only after Simplify returns. If the compiler stops in Simplify, nothing
   is said of the term before.

   The child prints, and ends with the status of the check ([Unix._exit], so
   that nothing that the compiler has registered to be done at the end is done
   in it, and no buffer of the compiler is written twice). The process that
   compiles waits for it and takes its status, and performs no check itself: the
   names that the check makes, the units that it loads and what it asks of
   [Compilenv] are in the child, not in the process in which the compiler goes
   on with the term after. (Until this was mended the check was made in the
   process that compiles: its names came before those that the compiler makes
   for what the .cmx file has, and asking for the units read left a section in
   that file; discrepancy V2-7 of middle_end/flambda2/rocq/DECISIONS.md.) A
   control compares, for some units compiled to the end, the files that the
   driver writes with those that the compiler writes (../../semantics/own_run/).

   What the child still shares with the process that compiles, and what is done
   about each. The positions of the open files: the child may read the .cmx file
   of another unit, which the compiler keeps open, and that moves the position
   of the file; the process that compiles takes the position of each of its
   descriptors that has one before the child is made, and puts back those that
   have moved when the child is known to have ended, but for the standard output
   and the standard error and any other descriptor of their files ([positions]
   below). The three standard descriptors are open before the compiler starts,
   one that is closed being given /dev/null, so that no file of the compiler
   takes the number of one. Where the descriptors cannot be listed no child is
   made, and the check is not made. The contents of files: the child writes on
   the standard output and the standard error and on nothing else, and it ends
   in [Unix._exit], so that no buffer of the compiler is written by it. The
   terminal, of which nothing is changed.

   Two things follow from the child. A limit of processor time is counted for it
   apart from the process that compiles: a unit can take up to twice the limit.
   And a signal that kills the check does not by itself stop compilation: the
   driver says that the check stopped and lets the compiler continue; it exits
   with 2 if compilation succeeds, and if compilation fails the compiler's
   status wins.

   What is printed is what the tool prints, under the name <unit>.fl, so that
   the two modes can be compared line by line. The conversion is the tool's, as
   it is: it sees in these terms what the text does not write (the kinds of the
   fields of a [Make_block], for one), and says "not checked" of what the model
   has not got.

   What is trusted here and not in the first mode: that the two hooks are given
   the terms that the compiler goes on with, and that opening any closed
   standard descriptors on /dev/null, retaining the first term, starting and
   waiting for the child, putting back the positions of the open files, and
   reporting its status do not change the compiler's compilation. What is not
   checked: the term after the reaper, where it is enabled (the hook
   [Reaped_flambda2]), and a unit on which the compiler does not run Simplify
   ([-Oclassic], or an attribute of the unit that asks for it), of which the
   driver says so. Of such a unit nothing is asked at all, not even whether its
   term is well-formed: the hook [Flambda2], which this driver uses, is not run
   for it, and after [reset_symbol_tables] (middle_end/flambda2/flambda2.ml) has
   cleared the tables of names the term that was kept cannot be converted. The
   line that says that Simplify was not run says nothing of the term. (A child
   made in the first hook could ask that question of every unit, and of a unit
   on which Simplify then stops; that is not done.)

   The units asked about are those whose .cmx file the compilation has read when
   Simplify returns ([Flsemantics.units_read]), as in the first mode. Which
   files the compilation finds is for the compiler's load path
   ([Compmisc.init_path]): the current directory unless [-nocwd] is given, the
   directories of [-I] and [-H], and the standard library unless [-nostdlib] is
   given. The first mode looks only in the directories of [-I]
   ([Flsemantics.init_load_path]).

   One unit is checked in a run of the driver, as in a run of the first mode.
   Given a second unit, the driver says so of it, checks nothing from then on,
   and exits with 2 where the compiler exits 0. (The first mode has a reason
   that the driver has no longer: the conversion keeps what it has noted of a
   unit, and here that is in a child that has ended. The rule is kept as it
   was.)

   The driver exits with the status of the compiler where that is not 0. Where
   it is 0, the driver exits with the status of the check: 0; 1 if the check
   failed (as the first mode does: the term after is not well-formed, or the two
   runs differ); 2 if the check raised an exception, which is printed; if the
   check was not made because the standard descriptors could not be opened, the
   descriptors could not be listed or no process could be made; if its process
   did not end with the status of a check, or is not known to have ended; if a
   position could not be put back; of each of which a line says so; or if a
   second unit was given. *)

(* Eta-expansion gives [create_process] and [waitpid] the unannotated types
   required by [Compiler_owee.Unix_intf.S] (as in driver/oxcaml_main.ml). *)
module Unix_for_owee = struct
  include Unix

  let create_process prog args stdin stdout stderr =
    Unix.create_process prog args stdin stdout stderr

  let waitpid flags pid = Unix.waitpid flags pid
end

(* The name under which the unit being compiled is shown: that of the file that
   the first mode would be given. *)
let shown () =
  match Env.get_current_unit () with
  | Some unit_info ->
    String.uncapitalize_ascii
      (Compilation_unit.name_as_string (Unit_info.modname unit_info))
    ^ ".fl"
  | None -> "(no unit).fl"

type state =
  | No_unit_yet
  | Given_to_simplify of
      { unit : Flambda2_terms.Flambda_unit.t;
        shown : string
      }
  | Checked
  | Refused

(* Where the run is. [Given_to_simplify] holds the term given to Simplify until
   Simplify has returned; nothing has been asked of the term. [Refused] is for
   good: a second unit has come in this process, and nothing is checked from
   then on. *)
let state = ref No_unit_yet

(* The status of the check: 0, 1 where it failed, 2 where it raised or was
   refused. *)
let status = ref 0

let guarded ~shown what f =
  try f ()
  with exn ->
    Format.printf "%s: the check stopped (%s): %s@." shown what
      (Printexc.to_string exn);
    status := max !status 2

let simplify_not_run shown =
  Format.printf
    "%s: not checked: the compiler did not run Simplify on the unit@." shown

(* Nothing here may open a binder of a term or make a name in any other way:
   Simplify has not run yet (the header). [shown] reads the name of the unit
   that is being compiled, and [Format.printf] prints. *)
let on_raw_flambda (unit : Flambda2_terms.Flambda_unit.t) =
  let shown = shown () in
  match !state with
  | No_unit_yet -> state := Given_to_simplify { unit; shown }
  | (Given_to_simplify _ | Checked | Refused) as earlier ->
    (* One unit to a run, as in the first mode (the header). Where Simplify was
       not run on the first unit its term is still held: it is let go here,
       before anything is printed, so that it is not compared with the term that
       Simplify returns for this unit. Nothing is asked of it: this unit has not
       been simplified yet. *)
    state := Refused;
    status := max !status 2;
    (match earlier with
    | Given_to_simplify { shown = first; _ } -> simplify_not_run first
    | No_unit_yet | Checked | Refused -> ());
    Format.printf
      "%s: not checked: a second unit in one process; run the driver once for \
       each unit@."
      shown

(* The two questions, as the tool asks them: whether the term that was kept is
   well-formed, and then the check of the two terms. Run in the child. *)
let ask ~shown unit after =
  let well_formed = ref false in
  guarded ~shown "the term given to Simplify" (fun () ->
      well_formed := Flsemantics.well_formed_before ~filename:shown unit);
  if !well_formed
  then
    guarded ~shown "the two terms" (fun () ->
        match
          Flsemantics.check ~filename:shown
            ~units:(Flsemantics.units_read ())
            unit after
        with
        | Failed -> status := max !status 1
        | Not_checked | Compared _ -> ())

external descriptor_of_int : int -> Unix.file_descr = "%identity"

external int_of_descriptor : Unix.file_descr -> int = "%identity"

(* The three standard descriptors are open before the compiler starts: one that
   is closed is given /dev/null. If the driver were started with one of them
   closed, the first file that the compiler opens would take its number, and a
   file that the compiler reads would then be where the driver prints, or be
   left out of [positions] as a place where it prints. A new descriptor is the
   lowest that is free, so /dev/null is opened until a descriptor above 2 comes,
   which is closed. Gives the error if /dev/null could not be opened: no check
   is made then ([in_a_child]). A descriptor is an integer on the systems that
   have [Unix.fork]. *)
let standard_descriptors_open () =
  let rec fill () =
    let descriptor = Unix.openfile "/dev/null" [Unix.O_RDWR] 0 in
    if int_of_descriptor descriptor <= 2 then fill () else Unix.close descriptor
  in
  match fill () with () -> None | exception exn -> Some exn

(* The descriptors of this process that have a position, each with its position,
   but those of the files on which the child prints. A child shares the open
   files of its parent, and with each its position: if the check reads a section
   of the .cmx file of another unit, which [File_sections] keeps open and reads
   when it is asked for, the position moves under this process too, whose
   channel does not know it. So the positions are taken before the child is made
   and put back when it has ended ([put_back]).

   Left out: the standard output and the standard error, and any other
   descriptor of their files (the same device and the same number of the file):
   the child prints there, and that position is to move. The standard input is
   not left out for its number. A descriptor that has no position (a pipe, a
   terminal) has none to take, and the one that the listing itself opens is
   closed by the time it is asked for its position.

   The descriptors are listed by the system (/proc/self/fd, or /dev/fd). *)
let positions () =
  let listed directory =
    List.filter_map int_of_string_opt (Array.to_list (Sys.readdir directory))
  in
  let descriptors =
    try listed "/proc/self/fd" with Sys_error _ -> listed "/dev/fd"
  in
  let file descriptor =
    match Unix.LargeFile.fstat descriptor with
    | { Unix.LargeFile.st_dev; st_ino; _ } -> Some (st_dev, st_ino)
    | exception Unix.Unix_error _ -> None
  in
  let printed_on = List.filter_map file [Unix.stdout; Unix.stderr] in
  List.filter_map
    (fun number ->
      let descriptor = descriptor_of_int number in
      match Unix.LargeFile.lseek descriptor 0L Unix.SEEK_CUR with
      | exception Unix.Unix_error _ -> None
      | position -> (
        match file descriptor with
        | None -> None
        | Some file when List.mem file printed_on -> None
        | Some _ -> Some (descriptor, position)))
    descriptors

(* Puts back each position that has moved, and touches no other; each descriptor
   is then where it was when its position was taken, which is asked of the
   system and not supposed. Gives the first error, if a position could not be
   asked for, or could not be put back. *)
let put_back positions =
  List.fold_left
    (fun error (descriptor, position) ->
      match
        if
          not
            (Int64.equal
               (Unix.LargeFile.lseek descriptor 0L Unix.SEEK_CUR)
               position)
        then
          if
            not
              (Int64.equal
                 (Unix.LargeFile.lseek descriptor position Unix.SEEK_SET)
                 position)
          then failwith "lseek gives another position than the one asked for"
      with
      | () -> error
      | exception exn -> (
        match error with None -> Some exn | Some _ -> error))
    None positions

(* Whether the three standard descriptors were open, or could be opened, when
   the driver started. *)
let standard_descriptors = ref None

(* Runs [f] in a child process and gives the status with which the child ends:
   that of the check, which [f] leaves in [status]. What this process has still
   to print is printed first, so that the child does not print it again, and the
   positions of its open files are taken ([positions]).

   The child ends in [Unix._exit] on every path, and no exception leaves it: if
   one did, the child would go on as a second compiler of the unit, or run what
   the compiler does on an error, which removes the files being written. So
   whatever is raised in the child is caught, [Stack_overflow] and
   [Out_of_memory] too, with the line that [guarded] prints and the status 2;
   and what could still be raised around that is caught where the child is made,
   with no line. A call of [exit] in the child would run what the compiler has
   registered with [at_exit]: nothing that [f] reaches calls it, and the child
   registers first a function that says so and ends the child, which is run
   before the compiler's. Before it ends the child flushes the two formatters
   and the two channels that it prints on, and nothing else: the compiler has
   other channels open (the assembly file, a file of dumps), and [Unix._exit]
   neither writes their buffers nor runs what the compiler registered with
   [at_exit].

   This process waits until the child is known to have ended, and then puts back
   the positions that have moved. [Unix.waitpid] is not asked for a child that
   is stopped; if it answers so, or fails otherwise than by an interruption or
   by saying that there is no such child, the child is killed and waited for
   again, so that it moves no position after they are put back. A child of which
   the system says that there is none has ended; its status is not known. If the
   child is still not known to have ended, the positions are put back as far as
   that can be done, and the line says that what is compiled from there is not
   to be trusted.

   Where the standard descriptors could not be opened, the descriptors could not
   be listed, or no child could be made, the check is not made in this process
   in its place: that is the fault that was mended. A line says so, and the
   status of the check is 2; and so where the child did not end with the status
   of a check (a signal killed it, or it ended with another status, or its
   status is not known), and where a position could not be put back, of which
   the line says that what is compiled from there is not to be trusted. The
   compiler goes on in each case. *)
let in_a_child ~shown f =
  let signal_shown signal =
    (* [Unix.waitpid] gives the numbers of [Sys], which are not the system's. *)
    if signal = Sys.sigkill
    then "SIGKILL"
    else if signal = Sys.sigxcpu
    then "SIGXCPU"
    else if signal = Sys.sigsegv
    then "SIGSEGV"
    else if signal = Sys.sigterm
    then "SIGTERM"
    else if signal = Sys.sigint
    then "SIGINT"
    else Printf.sprintf "the signal that [Sys] numbers %d" signal
  in
  let flush_all () =
    Format.pp_print_flush Format.std_formatter ();
    Format.pp_print_flush Format.err_formatter ();
    flush stdout;
    flush stderr
  in
  let stopped what how =
    Format.printf "%s: the check stopped (%s): %s@." shown what how;
    2
  in
  let not_to_be_trusted what =
    what ^ ": what is compiled from here is not to be trusted"
  in
  let in_child () =
    let of_the_check =
      try
        at_exit (fun () ->
            (try
               ignore
                 (stopped "the process of the check" "it called exit" : int)
             with _ -> ());
            (try flush_all () with _ -> ());
            Unix._exit 2);
        f ();
        !status
      with exn -> (
        try stopped "the process of the check" (Printexc.to_string exn)
        with _ -> 2)
    in
    (try flush_all () with _ -> ());
    Unix._exit of_the_check
  in
  let ended child =
    let kill () = try Unix.kill child Sys.sigkill with _ -> () in
    let rec wait ~killed =
      match Unix.waitpid [] child with
      | _, WEXITED code -> `Exited code
      | _, WSIGNALED signal -> `Killed signal
      | _, WSTOPPED _ ->
        kill ();
        wait ~killed:true
      | exception Unix.Unix_error (Unix.EINTR, _, _) -> wait ~killed
      | exception Unix.Unix_error (Unix.ECHILD, _, _) -> `Gone
      | exception exn ->
        if killed
        then `Not_known exn
        else (
          kill ();
          wait ~killed:true)
    in
    wait ~killed:false
  in
  let in_parent child positions =
    match ended child with
    | `Not_known exn ->
      ignore (put_back positions : exn option);
      stopped
        (not_to_be_trusted "the process of the check is not known to have ended")
        (Printexc.to_string exn)
    | (`Exited _ | `Killed _ | `Gone) as how -> (
      let of_the_check =
        match how with
        | `Exited ((0 | 1 | 2) as of_the_check) -> of_the_check
        | `Exited other ->
          stopped "the process of the check"
            (Printf.sprintf "it ended with status %d" other)
        | `Killed signal ->
          stopped "the process of the check"
            (Printf.sprintf "killed by %s" (signal_shown signal))
        | `Gone ->
          stopped "the process of the check"
            "it has ended, and its status is not known"
      in
      match put_back positions with
      | None -> of_the_check
      | Some exn ->
        stopped
          (not_to_be_trusted "a position of an open file could not be put back")
          (Printexc.to_string exn))
  in
  match !standard_descriptors with
  | Some exn -> stopped "the standard descriptors" (Printexc.to_string exn)
  | None -> (
    flush_all ();
    match positions () with
    | exception exn ->
      stopped "the descriptors of this process" (Printexc.to_string exn)
    | positions -> (
      match Unix.fork () with
      | exception exn -> stopped "fork" (Printexc.to_string exn)
      | 0 -> ( try in_child () with _ -> Unix._exit 2)
      | child -> in_parent child positions))

(* Simplify has returned: the term that was kept is asked about first, as the
   tool asks of the term that it has read before it runs Simplify, and then the
   two terms are checked; both in a child process (the header). *)
let on_flambda (after : Flambda2_terms.Flambda_unit.t) =
  match !state with
  | No_unit_yet | Checked | Refused -> ()
  | Given_to_simplify { unit; shown } ->
    state := Checked;
    let of_the_check = in_a_child ~shown (fun () -> ask ~shown unit after) in
    status := max !status of_the_check

let () =
  standard_descriptors := standard_descriptors_open ();
  Compiler_hooks.register Raw_flambda2 on_raw_flambda;
  Compiler_hooks.register Flambda2 on_flambda;
  let compiler =
    Optmaindriver.main
      (module Unix_for_owee : Compiler_owee.Unix_intf.S)
      Sys.argv Format.err_formatter
      ~flambda2:(module Flambda2)
  in
  (match !state with
  | Given_to_simplify { shown; _ } when compiler = 0 -> simplify_not_run shown
  | No_unit_yet | Given_to_simplify _ | Checked | Refused -> ());
  exit (if compiler <> 0 then compiler else !status)
