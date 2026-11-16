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

   - "not checked: ...", 2: the comparison stopped, the compiler did not reach
   the middle end, or a second unit was given. If the compiler fails, the status
   is the compiler's.

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

let on_raw_flambda unit =
  let answer =
    try check unit with exn -> Not_checked (one_line (Printexc.to_string exn))
  in
  match answer with
  | The_term ->
    print_endline "the text is the term";
    exit 0
  | Not_the_term ->
    print_endline "the text is not the term";
    exit 1
  | Not_read_back why ->
    print_endline ("the text is not read back: " ^ why);
    exit 1
  | Not_checked why ->
    print_endline ("not checked: " ^ why);
    exit 2

let () =
  let argv =
    match Array.to_list Sys.argv with
    | program :: "-text-to" :: file :: rest ->
      text_to := Some file;
      Array.of_list (program :: rest)
    | _ -> Sys.argv
  in
  Compiler_hooks.register Raw_flambda2 on_raw_flambda;
  let compiler =
    Optmaindriver.main
      (module Unix_for_owee : Compiler_owee.Unix_intf.S)
      argv Format.err_formatter
      ~flambda2:(module Flambda2)
  in
  (* The hook stops the process, so it was not reached. *)
  if compiler = 0
  then (
    print_endline "not checked: the compiler did not reach the middle end";
    exit 2)
  else exit compiler
