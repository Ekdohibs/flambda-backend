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

(* The compiler, with one check: that the text of the term that it gives to
   Simplify, read back, is that term. It is the driver of the compiler
   (driver/oxcaml_main.ml) with one hook registered (driver/compiler_hooks.mli):
   it takes the command line of the compiler, and two options of its own, which
   come first if they are given:

   fltextisterm.exe [-text-to FILE] [-differences-to FILE] <the command line of
   the compiler>

   The hook [Raw_flambda2] is given the term that the compiler gives to Simplify
   (middle_end/flambda2/flambda2.ml, [flambda_to_flambda0], before anything else
   is done with the term). The first time it is called, the driver

   1. writes the term as text, as [-drawfexpr-to] does ([Flambda_to_fexpr.conv],
   [Print_fexpr.flambda_unit]): to FILE if [-text-to] is given, to a temporary
   file otherwise;

   2. reads that file with the reader of texts ([Parse_flambda.parse_fexpr],
   [Fexpr_to_flambda.conv]), for the compilation unit being compiled;

   3. compares the term in memory with the term read, with the comparison of the
   compiler ([Compare.flambda_units_and_differences]; ../fldiff.ml uses
   [Compare.flambda_units]);

   4. keeps the answer, and lets the compiler go on.

   The hook counts its calls: the compiler calls it once for each implementation
   that it compiles as far as the middle end, however the implementations are
   given to it. When the compiler has returned, the driver prints one line on
   standard output and exits:

   - "the text is the term", 0: the hook was called once, and the comparison
   said that the two are equivalent;

   - "the text is not the term: WHAT", 1: called once, and the comparison said
   that they are different; WHAT is the name that it gives to the first thing
   that it found them to differ in (a fixed string, with the name of a primitive
   for a primitive; see [Env.note_difference] in
   middle_end/flambda2/compare/compare.ml). With [-differences-to], the names of
   all the differences that it found are written to FILE, one on a line, each
   once, in the order in which they were found; the file is empty if the text is
   the term;

   - "the text is not read back: ...", 1: called once, and the reader refused
   the text, or stopped, in whichever of its two steps;

   - "not checked: ...", 2: the printer stopped, so that there is no text (the
   term has something that it does not write), or the comparison stopped; or the
   hook was called more than once (more than one implementation was given: the
   files of [-text-to] and [-differences-to] are then removed); or it was not
   called (the compiler did not reach the middle end: an interface alone, [-i],
   [-stop-after typing]).

   If the compiler fails and returns to the driver, the driver prints no line
   and exits with the compiler's status. If the compiler exits by itself, with
   whatever status, as it does for the options that make it print something and
   stop ([-config], [-depend], ...) and for some errors of the command line, the
   driver prints "not checked: the compiler exited by itself" and exits with 2.

   So the driver exits with 0 only after it has printed "the text is the term",
   and it prints that only if the compiler compiled one implementation as far as
   the middle end and returned with 0.

   The compiler goes on after the check as it would alone: give [-stop-after
   middle_end] to stop it there, as the scripts of the tests do. It then leaves
   what the compiler leaves when it is stopped there (X.cmi if X has no
   interface, X.cmx). The driver is not for building: the reader of texts has
   made names in the compiler's process, and what the compiler writes afterwards
   need not be byte for byte what it writes alone.

   What is trusted: that the hook is given the term that the compiler goes on
   with, and the comparison. What the comparison does not read of a term is not
   checked (see middle_end/flambda2/compare/compare.ml). *)

(* Eta-expansion gives [create_process] and [waitpid] the unannotated types
   required by [Compiler_owee.Unix_intf.S] (as in driver/oxcaml_main.ml). *)
module Unix_for_owee = struct
  include Unix

  let create_process prog args stdin stdout stderr =
    Unix.create_process prog args stdin stdout stderr

  let waitpid flags pid = Unix.waitpid flags pid
end

module P = Flambda2_parser

let text_to = ref None

let differences_to = ref None

let one_line s = String.concat " " (String.split_on_char '\n' s)

let position (loc : Location.t) =
  Printf.sprintf "line %d, character %d" loc.loc_start.pos_lnum
    (loc.loc_start.pos_cnum - loc.loc_start.pos_bol)

let current_unit () =
  match Env.get_current_unit () with
  | Some unit_info -> Unit_info.modname unit_info
  | None -> Misc.fatal_error "No compilation unit"

type answer =
  | The_term
  | Not_the_term of string list
  | Not_read_back of string
  | Not_checked of string

let check (unit : Flambda2_terms.Flambda_unit.t) =
  let file, temporary =
    match !text_to with
    | Some file -> file, false
    | None -> Filename.temp_file "fltextisterm" ".fl", true
  in
  Misc.try_finally
    ~always:(fun () -> if temporary then Misc.remove_file file)
    (fun () ->
      let oc = open_out file in
      let ppf = Format.formatter_of_out_channel oc in
      Flambda2_ui.Flambda_colours.without_colours ~f:(fun () ->
          P.Print_fexpr.flambda_unit ppf (P.Flambda_to_fexpr.conv unit);
          Format.pp_print_flush ppf ());
      close_out oc;
      match P.Parse_flambda.parse_fexpr file with
      | exception exn -> Not_read_back (one_line (Printexc.to_string exn))
      | Error (Parsing_error (_, loc)) ->
        Not_read_back ("syntax error at " ^ position loc)
      | Error (Lexing_error (_, loc)) ->
        Not_read_back ("lexical error at " ^ position loc)
      | Ok fexpr -> (
        match P.Fexpr_to_flambda.conv (current_unit ()) fexpr with
        | exception exn -> Not_read_back (one_line (Printexc.to_string exn))
        | { unit = read; code_slot_offsets = _ } -> (
          match
            Flambda2_compare.Compare.flambda_units_and_differences unit read
          with
          | exception exn -> Not_checked (one_line (Printexc.to_string exn))
          | Equivalent, _ -> The_term
          | Different { approximant = _ }, differences ->
            Not_the_term differences)))

(* Each name once, in the order in which the differences were found. *)
let write_differences differences =
  match !differences_to with
  | None -> ()
  | Some file ->
    let oc = open_out file in
    let _ : string list =
      List.fold_left
        (fun seen difference ->
          if List.mem difference seen
          then seen
          else (
            output_string oc (difference ^ "\n");
            difference :: seen))
        [] differences
    in
    close_out oc

(* What the check said of the first implementation, and how many times the hook
   has been called: once for each implementation that the compiler compiles as
   far as the middle end. *)
let first_answer = ref None

let calls = ref 0

let on_raw_flambda unit =
  incr calls;
  if !calls = 1
  then
    first_answer
      := Some
           (try check unit
            with exn -> Not_checked (one_line (Printexc.to_string exn)))

(* Whether the driver has printed its line, or exits with the status that the
   compiler returned. *)
let answered = ref false

let answer line status =
  answered := true;
  print_endline line;
  exit status

(* For some options, and for some errors, the compiler exits by itself, without
   returning to the driver. This function is registered when this module is
   initialised: at the exit of the process it is run after the functions that
   the compiler registers as it runs, and before those that the libraries
   registered when they were initialised. If the process is exiting and the
   driver has not answered, it says so, and exits with 2 in the place of the
   compiler's status: that second exit runs the functions that remain. *)
let () =
  at_exit (fun () ->
      if not !answered
      then (
        (* On a line of its own, whatever the compiler printed. *)
        print_newline ();
        answer "not checked: the compiler exited by itself" 2))

let () =
  let rec own_options args =
    match args with
    | "-text-to" :: file :: args ->
      text_to := Some file;
      own_options args
    | "-differences-to" :: file :: args ->
      differences_to := Some file;
      own_options args
    | _ -> args
  in
  let argv =
    Array.of_list
      (Sys.argv.(0) :: own_options (List.tl (Array.to_list Sys.argv)))
  in
  Compiler_hooks.register Raw_flambda2 on_raw_flambda;
  let compiler =
    Optmaindriver.main
      (module Unix_for_owee : Compiler_owee.Unix_intf.S)
      argv Format.err_formatter
      ~flambda2:(module Flambda2)
  in
  if compiler <> 0
  then (
    answered := true;
    exit compiler)
  else if !calls > 1
  then (
    (* The text written is that of the first implementation only. *)
    Option.iter Misc.remove_file !text_to;
    Option.iter Misc.remove_file !differences_to;
    answer "not checked: more than one implementation is given" 2)
  else
    (* The hook sets [first_answer] at its first call. *)
    match !first_answer with
    | None -> answer "not checked: the compiler did not reach the middle end" 2
    | Some The_term ->
      write_differences [];
      answer "the text is the term" 0
    | Some (Not_the_term differences) ->
      write_differences differences;
      answer
        ("the text is not the term: "
        ^ match differences with first :: _ -> first | [] -> "(no name)")
        1
    | Some (Not_read_back why) -> answer ("the text is not read back: " ^ why) 1
    | Some (Not_checked why) -> answer ("not checked: " ^ why) 2
