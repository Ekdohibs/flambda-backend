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
   it takes the command line of the compiler, and one option of its own, which
   comes first if it is given:

   fltextisterm.exe [-text-to FILE] <the command line of the compiler>

   The hook [Raw_flambda2] is given the term that the compiler gives to Simplify
   (middle_end/flambda2/flambda2.ml, [flambda_to_flambda0], before anything else
   is done with the term). The driver then

   1. writes the term as text, as [-drawfexpr-to] does ([Flambda_to_fexpr.conv],
   [Print_fexpr.flambda_unit]): to FILE if [-text-to] is given, to a temporary
   file otherwise;

   2. reads that file with the reader of texts ([Parse_flambda.parse_fexpr],
   [Fexpr_to_flambda.conv]), for the compilation unit being compiled;

   3. compares the term in memory with the term read, with the comparison of the
   compiler ([Compare.flambda_units], as ../fldiff.ml does);

   4. prints one line on standard output and stops the process: Simplify is not
   run and no .cmx file is written.

   The line and the exit status:

   - "the text is the term", 0: the comparison says that the two are equivalent;

   - "the text is not the term", 1: it says that they are different;

   - "the text is not read back: ...", 1: the reader refused the text, or
   stopped;

   - "not checked: ...", 2: the comparison stopped; more than one implementation
   was given; the compiler did not reach the middle end (an interface alone), or
   exited before it by itself, as it does for the options that make it print
   something and stop ([-config], [-version], [-depend], [-help]).

   If the compiler fails, the driver prints no line and exits with the
   compiler's status.

   The answer is the line. The driver exits with 0 only after it has printed
   "the text is the term"; a caller should ask for that line, the last of
   standard output, and not for the status alone.

   One implementation is checked in a process: the hook stops the process at the
   first unit, so a second would not be checked. The driver therefore refuses,
   before the compiler runs, a command line with more than one argument that
   ends in ".ml" or follows [-impl], and one with [-args] or [-args0], which
   read more arguments from a file. An interface is not counted.

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
  | Not_the_term
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
      | Error (Parsing_error (_, loc)) ->
        Not_read_back ("syntax error at " ^ position loc)
      | Error (Lexing_error (_, loc)) ->
        Not_read_back ("lexical error at " ^ position loc)
      | Ok fexpr -> (
        match P.Fexpr_to_flambda.conv (current_unit ()) fexpr with
        | exception exn -> Not_read_back (one_line (Printexc.to_string exn))
        | { unit = read; code_slot_offsets = _ } -> (
          match Flambda2_compare.Compare.flambda_units unit read with
          | exception exn -> Not_checked (one_line (Printexc.to_string exn))
          | Equivalent -> The_term
          | Different { approximant = _ } -> Not_the_term)))

(* Whether the driver has printed its line, or the compiler has returned to
   it. *)
let answered = ref false

let answer line status =
  answered := true;
  print_endline line;
  exit status

(* For some options the compiler prints something and exits by itself, with the
   status 0, without returning to the driver. This function is registered before
   anything that the compiler registers, so it is run after: if the process is
   exiting and the driver has not answered, it says so and exits with 2 in the
   place of the compiler's status. *)
let () =
  at_exit (fun () ->
      if not !answered
      then (
        Format.pp_print_flush Format.std_formatter ();
        Format.pp_print_flush Format.err_formatter ();
        (* On a line of its own, whatever the compiler printed. *)
        print_newline ();
        print_endline "not checked: the compiler exited before the middle end";
        flush stdout;
        flush stderr;
        Unix._exit 2))

let on_raw_flambda unit =
  match
    try check unit with exn -> Not_checked (one_line (Printexc.to_string exn))
  with
  | The_term -> answer "the text is the term" 0
  | Not_the_term -> answer "the text is not the term" 1
  | Not_read_back why -> answer ("the text is not read back: " ^ why) 1
  | Not_checked why -> answer ("not checked: " ^ why) 2

(* The implementations that a command line gives, counted generously: an
   argument of another option that ends in ".ml" is counted too. *)
let rec implementations args =
  match args with
  | [] -> 0
  | "-impl" :: _ :: args -> 1 + implementations args
  | arg :: args ->
    (if Filename.check_suffix arg ".ml" then 1 else 0) + implementations args

let () =
  let argv =
    match Array.to_list Sys.argv with
    | program :: "-text-to" :: file :: rest ->
      text_to := Some file;
      Array.of_list (program :: rest)
    | _ -> Sys.argv
  in
  let args = List.tl (Array.to_list argv) in
  if List.exists (fun arg -> List.mem arg ["-args"; "-args0"]) args
  then answer "not checked: -args and -args0 are not accepted" 2;
  if implementations args > 1
  then answer "not checked: more than one implementation is given" 2;
  Compiler_hooks.register Raw_flambda2 on_raw_flambda;
  let compiler =
    Optmaindriver.main
      (module Unix_for_owee : Compiler_owee.Unix_intf.S)
      argv Format.err_formatter
      ~flambda2:(module Flambda2)
  in
  (* The hook stops the process, so it was not reached. *)
  if compiler = 0
  then answer "not checked: the compiler did not reach the middle end" 2
  else (
    answered := true;
    exit compiler)
