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

(* The program of the test that what Simplify makes of a unit does not depend on
   the order in which the compiler holds the names of the unit
   (../../semantics/name_order/ has the rules and says what is tested). It does
   one of three things, one unit to a process:

   flnameorder.exe -simplify-to RESULT.fl [-burn K] <options of the compiler>
   UNIT.fl

   reads the text UNIT.fl with the reader of texts ([Parse_flambda.parse_fexpr],
   [Fexpr_to_flambda.conv]), gives the unit that the reader made, as it is, to
   Simplify ([Flambda2.flambda_to_flambda], for a machine of 64 bits), and
   writes the unit that Simplify returns as text to RESULT.fl
   ([Flambda_to_fexpr.conv], [Print_fexpr.flambda_unit]). Before the text is
   converted, K variables and K continuations are made and thrown away
   ([Variable.create], [Continuation.create]); K is 0 if [-burn] is not given. A
   variable and a continuation each take the next number of a counter when they
   are made, and their maps and sets ([Variable.Map], [Variable.Set], and those
   of [Continuation]) hold them in the order of their identifiers; the
   identifier is a hash of the unit and of that number, or the next one that is
   free where another name has that hash
   (middle_end/flambda2/identifiers/int_ids.ml, continuation.ml;
   middle_end/flambda2/algorithms/table_by_int_id.ml). So the names that the
   reader makes for the text have their numbers moved by K, which can change
   their order in those maps and sets (it need not, for any two names); the unit
   is the same unit. Which of two names of the text was made first is not
   changed, and no other order of making is tried; nor is a map that keeps the
   order in which it was filled ([Lmap]) moved.

   It prints one line and exits: "simplified", 0; "not read: WHY", 3, if the
   reader refuses the text or stops; "not simplified: WHY", 4, if Simplify stops
   with an exception; "not written: WHY", 6, if the printer of texts stops on
   the result. Simplify looks for the .cmx of another unit in the directories
   given with [-I] and in no other, as it does in ../flsemantics.ml
   ([init_load_path]). Unlike the check of that file, this program does not
   apply the checks that the check makes of a unit before Simplify, nor the
   adjustments of [as_read_from_text]: it gives Simplify the unit as
   ../fexprc.ml does.

   flnameorder.exe -compare FIRST.fl SECOND.fl

   reads two texts, both as units of the compilation unit that the name of the
   first file gives, and compares them with the comparison of the compiler
   ([Compare.flambda_units_and_differences]), which pairs the variables, the
   continuations, the symbols and the code that the two units bind, and asks of
   a symbol and of code that neither binds that it be the same in both. It
   prints one line and exits: "the same", 0; "differs: WHAT", 1, where WHAT is
   the name that the comparison gives to the first thing that it found the two
   to differ in; "not compared: WHY", 2, if the reader refuses one of the texts
   or stops, or the comparison stops.

   flnameorder.exe -order K

   makes K variables and K continuations and throws them away, as above, then
   makes eight variables and eight continuations, and prints two lines: the
   eight variables, each by its place among the eight in the order in which they
   were made (0 to 7), in the order in which a set of the compiler holds them
   ([Variable.Set.elements]); and the same for the continuations. It is a
   control of the test, on a sample: that the lines differ for two numbers K
   shows that K changes the order of these eight variables and of these eight
   continuations, and not that it changes an order in every unit. If the lines
   come to be the same, the control no longer shows that names are moved.

   What is trusted: the comparison, and that the text of a unit says what the
   unit is. What the text does not write of a unit, and what the comparison does
   not read (middle_end/flambda2/compare/compare.ml), is not compared. *)

module P = Flambda2_parser
module Flambda_kind = Flambda2_kinds.Flambda_kind
module Variable = Flambda2_identifiers.Variable
module Continuation = Flambda2_identifiers.Continuation

let answer line status =
  print_endline line;
  exit status

let one_line s = String.concat " " (String.split_on_char '\n' s)

let position (loc : Location.t) =
  Printf.sprintf "line %d, character %d" loc.loc_start.pos_lnum
    (loc.loc_start.pos_cnum - loc.loc_start.pos_bol)

let parse file =
  match P.Parse_flambda.parse_fexpr file with
  | Error (Parsing_error (_, loc)) -> Error ("syntax error at " ^ position loc)
  | Error (Lexing_error (_, loc)) -> Error ("lexical error at " ^ position loc)
  | Ok fexpr -> Ok fexpr
  | exception exn -> Error (one_line (Printexc.to_string exn))

let convert comp_unit fexpr =
  match P.Fexpr_to_flambda.conv comp_unit fexpr with
  | exception exn -> Error (one_line (Printexc.to_string exn))
  | result -> Ok result

(* The names that are made and thrown away. *)
let burn stamps =
  for _ = 1 to stamps do
    let (_ : Variable.t) = Variable.create "burnt" Flambda_kind.value in
    let (_ : Continuation.t) = Continuation.create () in
    ()
  done

let order ~burnt =
  Compilenv.reset (P.Parse_flambda.make_unit_info ~filename:"order.fl");
  burn burnt;
  let made make = List.init 8 (fun place -> make (), place) in
  let show what ~elements ~equal made =
    let places =
      List.map
        (fun name ->
          let _, place = List.find (fun (made, _) -> equal name made) made in
          string_of_int place)
        (elements (List.map fst made))
    in
    Printf.printf "%s: %s\n" what (String.concat " " places)
  in
  show "variables"
    ~elements:(fun names -> Variable.Set.elements (Variable.Set.of_list names))
    ~equal:Variable.equal
    (made (fun () -> Variable.create "made" Flambda_kind.value));
  show "continuations"
    ~elements:(fun names ->
      Continuation.Set.elements (Continuation.Set.of_list names))
    ~equal:Continuation.equal
    (made (fun () -> Continuation.create ()));
  exit 0

(* As [init_load_path] of ../flsemantics.ml: the directories of [-I], and no
   other. *)
let init_load_path () =
  Load_path.reset ();
  List.iter
    (fun ({ path; cmx_guaranteed } : Clflags.visible_include) ->
      Load_path.add_dir (Visible { cmx_guaranteed }) path)
    (List.rev !Clflags.include_dirs);
  Env.reset_cache ~preserve_persistent_env:false

let write_unit file unit =
  let oc = open_out file in
  let ppf = Format.formatter_of_out_channel oc in
  Flambda2_ui.Flambda_colours.without_colours ~f:(fun () ->
      P.Print_fexpr.flambda_unit ppf (P.Flambda_to_fexpr.conv unit);
      Format.pp_print_flush ppf ());
  close_out oc

let simplify ~burnt ~result file =
  match parse file with
  | Error why -> answer ("not read: " ^ why) 3
  | Ok fexpr -> (
    let unit_info = P.Parse_flambda.make_unit_info ~filename:file in
    (* As the driver does for a unit that it compiles (and ../flsemantics.ml):
       it sets the compilation unit, which the making of a name asks for. *)
    Compilenv.reset unit_info;
    burn burnt;
    match convert (Unit_info.modname unit_info) fexpr with
    | Error why -> answer ("not read: " ^ why) 3
    | Ok { unit; code_slot_offsets } -> (
      let null_formatter =
        Format.make_formatter (fun _ _ _ -> ()) (fun () -> ())
      in
      match
        Flambda2.flambda_to_flambda ~ppf_dump:null_formatter
          ~prefixname:(Filename.chop_extension file)
          ~machine_width:Sixty_four ~code_slot_offsets unit
      with
      | exception exn ->
        answer ("not simplified: " ^ one_line (Printexc.to_string exn)) 4
      | simplified -> (
        match write_unit result simplified.flambda with
        | exception exn ->
          answer ("not written: " ^ one_line (Printexc.to_string exn)) 6
        | () -> answer "simplified" 0)))

let compare first second =
  let unit_info = P.Parse_flambda.make_unit_info ~filename:first in
  Compilenv.reset unit_info;
  let read which file =
    match Result.bind (parse file) (convert (Unit_info.modname unit_info)) with
    | Error why ->
      answer
        (Printf.sprintf "not compared: the %s text is not read back: %s" which
           why)
        2
    | Ok { unit; code_slot_offsets = _ } -> unit
  in
  let first = read "first" first in
  let second = read "second" second in
  match Flambda2_compare.Compare.flambda_units_and_differences first second with
  | exception exn ->
    answer ("not compared: " ^ one_line (Printexc.to_string exn)) 2
  | Equivalent, _ -> answer "the same" 0
  | Different { approximant = _ }, differences ->
    answer
      ("differs: "
      ^ match differences with first :: _ -> first | [] -> "(no name)")
      1

module Options = Oxcaml_args.Make_optcomp_options (Oxcaml_args.Default.Optmain)

let usage () =
  prerr_endline
    "Usage: flnameorder.exe -simplify-to RESULT.fl [-burn K] <options of the \
     compiler> UNIT.fl\n\
    \       flnameorder.exe -compare FIRST.fl SECOND.fl\n\
    \       flnameorder.exe -order K";
  exit 5

let number_of_names s =
  match int_of_string_opt s with
  | Some stamps when stamps >= 0 -> stamps
  | Some _ | None -> usage ()

let () =
  match List.tl (Array.to_list Sys.argv) with
  | ["-compare"; first; second] ->
    Clflags.set_target Clflags.Target.Native;
    compare first second
  | ["-order"; stamps] ->
    Clflags.set_target Clflags.Target.Native;
    order ~burnt:(number_of_names stamps)
  | "-simplify-to" :: result :: args -> (
    let burnt, args =
      match args with
      | "-burn" :: stamps :: args -> number_of_names stamps, args
      | _ -> 0, args
    in
    (* Calling [Clflags.set_target Native] is how we tell the option parsing
       machinery that we are going to use flambda2 (as ../fexprc.ml). *)
    Clflags.set_target Clflags.Target.Native;
    let files = ref [] in
    let defer_file file =
      match Filename.extension file with
      | ".fl" -> files := file :: !files
      | ext -> Misc.fatal_errorf "Unrecognized extension %s" ext
    in
    Compenv.warnings_for_discarded_params := true;
    Compenv.set_extra_params (Some Oxcaml_args.Extra_params.read_param);
    Compenv.readenv Format.err_formatter Before_args;
    Clflags.add_arguments __LOC__ (Arch.command_line_options @ Options.list);
    Clflags.Opt_flag_handler.set Oxcaml_flags.opt_flag_handler;
    Compenv.parse_arguments
      (ref (Array.of_list (Sys.argv.(0) :: args)))
      defer_file "flnameorder";
    Location.read_clflags_from_env ();
    init_load_path ();
    match !files with
    | [file] -> simplify ~burnt ~result file
    | [] | _ :: _ :: _ -> usage ())
  | _ -> usage ()
