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

(* A check of the conversion of flsemantics.ml, which is trusted: a round trip
   through fexpr.

   The unit of an [.fl] file is converted to a term of the Rocq model. That term
   is written back as fexpr (flmodel_to_fexpr.ml), the text is read again by the
   compiler's parser, and the unit read is converted in its turn, with the names
   numbered afresh. The two terms of the model should be the same. They are
   compared by the model's own comparison (FlambdaCompareSpec.expr_eqb), which
   is up to the names of bound variables and continuations and asks symbols,
   code ids and slots to be the same; so the symbols, code ids and slots of the
   second term are first given the numbers they had in the first, which are read
   off the names that the printer gave them. A name is read as a number only if
   it is exactly what the printer makes of that number. And the value slots of
   each set of closures are put, in both terms, in the order of their numbers
   ([slots_in_order] says why).

   The same is done of the unit that Simplify makes of the one in the file,
   whose conversion is trusted as much, and which has what the parser of fexpr
   never makes of a file: so it is there that the limits of fexpr show.

   One answer is printed for the unit of the file, and one for the unit after
   Simplify: - "round trip the same"; - "round trip DIFFERS", with the first
   line at which the two terms, printed, differ. This is a mistake of the
   conversion or of the printer, something that the conversion drops and should
   not, or something that fexpr writes and its parser reads differently; - "not
   printable", with the reason: the term has something that fexpr cannot write,
   or that the printer does not cover; - "not read back": the compiler's parser,
   or its conversion of fexpr to Flambda, fails on the text printed, with an
   error that it reports or with an exception, or the text printed is one that
   the normalisation below refuses; - "names not read back", with the reason:
   the term read back uses a symbol, a code id or a slot whose name is not one
   that the printer makes, or two of them have one name or one number. This is a
   mistake of the printer or of the conversion; - "not converted", with the
   reason: the unit of the file has something that the model has not got, or its
   text is one that the normalisation below refuses; "not converted back" is the
   same of the unit read back, and is a mistake of the printer.

   A unit read from a text is first normalised as the check of a unit normalises
   it (flsemantics.ml, [with_lost_kinds]): each [Opaque_identity] whose kind as
   read is not the kind of its argument is given the kind of its argument, and a
   direct call or a call of unknown arity whose arity as read is one value for
   each argument may be given another ([with_lost_kinds] says which, and from
   what). That is done to the unit of the file and to each unit read back. So
   the round trip shows that the conversion and the printer agree once both
   units are so normalised; where the text of the file left out a kind or an
   arity that the term had, it does not show that.

   What this does not check: the kinds of the value slots, which are not in the
   term; the order of the value slots of a set of closures, which the text
   cannot keep; what the conversion does to the two units of a check together
   (the numbers it gives the code ids of the unit after: flsemantics.ml,
   [convert_both]); and what the conversion drops, when the printer puts nothing
   in its place: a mistake by which both terms lack the same thing is not seen.
   So it is a check that what the conversion makes of each example is stable
   under this round trip, in which the printer is a second reading of the model.
   It does not show that the conversion is one to one (a conversion that takes
   two things to one, with a printer that writes one of them, passes) nor that
   it keeps enough.

   This module does not know the conversion, nor how Simplify is run:
   flsemantics.ml gives both as functions. *)

open Import
module Kinds = Flambda2_rocq_semantics.FlambdaKindModel
module Model = Flambda2_rocq_semantics.FlambdaModel
module Compare = Flambda2_rocq_semantics.FlambdaCompareSpec
module Numbers = Flambda2_rocq_library.BinNums
module Printer = Flmodel_to_fexpr

type converted =
  { body : Model.expr;
    return_continuation : Model.continuation;
    exn_continuation : Model.continuation
  }

(* The conversion, as flsemantics.ml gives it. [convert] answers the reason when
   the unit has something that the model has not got. [as_read] is what the
   check of a unit does to a unit read from a text before anything else (it
   gives each [Opaque_identity] whose kind as read is not the kind of its
   argument the kind of its argument, and may give an arity to a direct call or
   a call of unknown arity whose arity as read is one value for each argument),
   or the reason why such a unit is not checked; it is applied to the unit of
   the file and to each unit read back, and not to the unit that Simplify makes,
   which is not read from a text. [reset] forgets every name numbered so far.
   The four lists are the global names numbered since the last [reset], each
   with its number. *)
type conversion =
  { convert : Flambda_unit.t -> (converted, string) result;
    as_read : Flambda_unit.t -> (Flambda_unit.t, string) result;
    reset : unit -> unit;
    value_slot_kind : Model.value_slot -> Kinds.flambda_kind option;
    symbols : unit -> (string * Numbers.positive) list;
    code_ids : unit -> (string * Numbers.positive) list;
    value_slots : unit -> (string * Numbers.positive) list;
    function_slots : unit -> (string * Numbers.positive) list
  }

(* ---- The global names of a term, renamed ------------------------------- *)

type renaming =
  { symbol : Model.symbol -> Model.symbol;
    code_id : Model.code_id -> Model.code_id;
    value_slot : Model.value_slot -> Model.value_slot;
    function_slot : Model.function_slot -> Model.function_slot
  }

let rename_simple r (s : Model.simple) : Model.simple =
  match s with Symbol s -> Symbol (r.symbol s) | Var _ | Const _ -> s

let rename_simples r = List.map (rename_simple r)

let rename_value_slots r slots =
  List.map (fun (slot, s) -> r.value_slot slot, rename_simple r s) slots

let rename_function_decls r function_decls =
  List.map
    (fun (slot, code) -> r.function_slot slot, r.code_id code)
    function_decls

let rename_named r (d : Model.named) : Model.named =
  match d with
  | Simple s -> Simple (rename_simple r s)
  | Prim (p, args) -> Prim (p, rename_simples r args)
  | Set_of_closures (code, slots) ->
    Set_of_closures (r.code_id code, rename_value_slots r slots)
  | Closure_of_set (slot, function_decls, slots) ->
    Closure_of_set
      ( r.function_slot slot,
        rename_function_decls r function_decls,
        rename_value_slots r slots )
  | Project_value_slot (slot, closure) ->
    Project_value_slot (r.value_slot slot, rename_simple r closure)
  | Project_value_slot_from (project_from, slot, closure) ->
    Project_value_slot_from
      (r.function_slot project_from, r.value_slot slot, rename_simple r closure)
  | Project_function_slot (move_from, move_to, closure) ->
    Project_function_slot
      ( r.function_slot move_from,
        r.function_slot move_to,
        rename_simple r closure )

let rename_static_const r (const : Model.static_const) : Model.static_const =
  match const with
  | Static_const_set_of_closures (code, slots) ->
    Static_const_set_of_closures (r.code_id code, rename_value_slots r slots)
  | Static_const_closure_of_set (slot, closure_symbols, function_decls, slots)
    ->
    Static_const_closure_of_set
      ( r.function_slot slot,
        List.map
          (fun (slot, s) -> r.function_slot slot, r.symbol s)
          closure_symbols,
        rename_function_decls r function_decls,
        rename_value_slots r slots )
  | Static_const_block (tag, fields) ->
    Static_const_block (tag, rename_simples r fields)
  | Static_const_boxed_number contents ->
    Static_const_boxed_number (rename_simple r contents)
  | Static_const_immutable_string _ -> const
  | Static_const_immutable_value_array elements ->
    Static_const_immutable_value_array (rename_simples r elements)

let rename_apply_cont r (ac : Model.apply_cont) : Model.apply_cont =
  { ac with ac_args = rename_simples r ac.ac_args }

let rec rename_expr r (e : Model.expr) : Model.expr =
  match e with
  | Let (x, k, d, body) -> Let (x, k, rename_named r d, rename_expr r body)
  | Let_static (bindings, body) ->
    Let_static (List.map (rename_binding r) bindings, rename_expr r body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive (k, rename_handler r h, rename_expr r body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( List.map (fun (k, h) -> k, rename_handler r h) handlers,
        rename_expr r body )
  | Apply ap ->
    let call_kind : Model.call_kind =
      match ap.ap_call_kind with
      | Direct code -> Direct (r.code_id code)
      | Indirect_unknown_arity -> Indirect_unknown_arity
      | Indirect_known_arity codes ->
        Indirect_known_arity (Option.map (List.map r.code_id) codes)
    in
    Apply
      { ap with
        ap_callee = rename_simple r ap.ap_callee;
        ap_call_kind = call_kind;
        ap_args = rename_simples r ap.ap_args;
        ap_exn_extra_args =
          List.map
            (fun (s, kind) -> rename_simple r s, kind)
            ap.ap_exn_extra_args
      }
  | Apply_cont ac -> Apply_cont (rename_apply_cont r ac)
  | Switch (scrutinee, arms) ->
    Switch
      ( rename_simple r scrutinee,
        List.map (fun (i, ac) -> i, rename_apply_cont r ac) arms )
  | Invalid -> Invalid

and rename_handler r (Model.Handler (params, metadata, handler)) :
    Model.continuation_handler =
  Handler (params, metadata, rename_expr r handler)

and rename_binding r (binding : Model.static_binding) : Model.static_binding =
  match binding with
  | Bind_code (id, Code (params, my_closure, ret, exn, metadata, body)) ->
    Bind_code
      ( r.code_id id,
        Code (params, my_closure, ret, exn, metadata, rename_expr r body) )
  | Bind_deleted_code id -> Bind_deleted_code (r.code_id id)
  | Bind_symbol (s, const) ->
    Bind_symbol (r.symbol s, rename_static_const r const)

(* ---- The value slots of a set, in the order of their numbers ----------- *)

(* The model keeps the value slots of a set of closures as a list, in the order
   of the compiler's map of value slots, which is the order in which the
   compiler made the slots (flsemantics.ml, [convert_value_slots]). The text
   cannot keep that order: the parser of fexpr makes a slot where the text first
   names it, and that may be in the code of a function, before the set, where
   the code projects some of the slots and not others. The machine looks a slot
   up by its name, and what a lookup finds does not depend on where the slot
   stands among the others (two bindings of one slot keep their order here: the
   sort is stable). So the two terms are compared with the slots of each set in
   the order of their numbers; what each slot holds is compared as before.

   This is a choice of what the round trip tells apart, and no proof that two
   such terms are equivalent: a closure that is a value still holds its slots as
   a list, in the order of the term, and a value that a unit returns shows
   it. *)
let sort_value_slots slots =
  List.stable_sort
    (fun (slot, _) (slot', _) ->
      Int.compare (Printer.int_of_positive slot) (Printer.int_of_positive slot'))
    slots

let slots_in_order_named (d : Model.named) : Model.named =
  match d with
  | Set_of_closures (code, slots) ->
    Set_of_closures (code, sort_value_slots slots)
  | Closure_of_set (slot, function_decls, slots) ->
    Closure_of_set (slot, function_decls, sort_value_slots slots)
  | Simple _ | Prim _ | Project_value_slot _ | Project_value_slot_from _
  | Project_function_slot _ ->
    d

let slots_in_order_static_const (const : Model.static_const) :
    Model.static_const =
  match const with
  | Static_const_set_of_closures (code, slots) ->
    Static_const_set_of_closures (code, sort_value_slots slots)
  | Static_const_closure_of_set (slot, closure_symbols, function_decls, slots)
    ->
    Static_const_closure_of_set
      (slot, closure_symbols, function_decls, sort_value_slots slots)
  | Static_const_block _ | Static_const_boxed_number _
  | Static_const_immutable_string _ | Static_const_immutable_value_array _ ->
    const

let rec slots_in_order (e : Model.expr) : Model.expr =
  match e with
  | Let (x, k, d, body) ->
    Let (x, k, slots_in_order_named d, slots_in_order body)
  | Let_static (bindings, body) ->
    Let_static (List.map slots_in_order_binding bindings, slots_in_order body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive (k, slots_in_order_handler h, slots_in_order body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( List.map (fun (k, h) -> k, slots_in_order_handler h) handlers,
        slots_in_order body )
  | Apply _ | Apply_cont _ | Switch _ | Invalid -> e

and slots_in_order_handler (Model.Handler (params, metadata, handler)) :
    Model.continuation_handler =
  Handler (params, metadata, slots_in_order handler)

and slots_in_order_binding (binding : Model.static_binding) :
    Model.static_binding =
  match binding with
  | Bind_code (id, Code (params, my_closure, ret, exn, metadata, body)) ->
    Bind_code
      (id, Code (params, my_closure, ret, exn, metadata, slots_in_order body))
  | Bind_deleted_code _ -> binding
  | Bind_symbol (s, const) -> Bind_symbol (s, slots_in_order_static_const const)

exception Names_not_read_back of string

let names_not_read_back fmt =
  Format.kasprintf (fun reason -> raise (Names_not_read_back reason)) fmt

let rec positive_of_int n : Numbers.positive =
  if n < 1
  then invalid_arg "Flroundtrip.positive_of_int"
  else if n = 1
  then Coq_xH
  else if n land 1 = 0
  then Coq_xO (positive_of_int (n lsr 1))
  else Coq_xI (positive_of_int (n lsr 1))

(* The renaming of one sort of name: a name numbered [m] the second time, and
   that the printer made from the number [n], goes to [n].

   Every global name that the term read back uses should be one that the printer
   made from a number, and the printer makes one name of one number. So it is a
   failure of the round trip, raised as [Names_not_read_back], that the term
   read back uses a name that is not exactly what the printer makes of a number,
   or a number that has no name; and that two names that the printer makes are
   numbered alike the second time, or one such name is numbered twice. A name
   that the printer did not make and the term does not use is not a failure: the
   conversion numbers the function slots of a unit whose sets of closures it
   then writes without them, and the parser names those after the code. *)
let back_to_numbers ~sort ~prefix (numbered : (string * Numbers.positive) list)
    =
  let table =
    List.fold_left
      (fun table (name, m) ->
        let m = Printer.int_of_positive m in
        let n = Printer.number_of_name ~prefix name in
        List.iter
          (fun (name', m', n') ->
            if m' = m
            then
              names_not_read_back "the %ss %s and %s are given one number" sort
                name' name;
            match n, n' with
            | Some n, Some n' when n = n' ->
              names_not_read_back "two %ss have the name %s" sort name
            | (Some _ | None), (Some _ | None) -> ())
          table;
        (name, m, n) :: table)
      [] numbered
  in
  fun m ->
    let m = Printer.int_of_positive m in
    match List.find_opt (fun (_, m', _) -> m' = m) table with
    | Some (_, _, Some n) -> positive_of_int n
    | Some (name, _, None) ->
      names_not_read_back "the %s %s is not a name that the printer makes" sort
        name
    | None -> names_not_read_back "a %s that has no name" sort

let renaming_of (conversion : conversion) : renaming =
  { symbol =
      back_to_numbers ~sort:"symbol" ~prefix:Printer.symbol_prefix
        (conversion.symbols ());
    code_id =
      back_to_numbers ~sort:"code id" ~prefix:Printer.code_id_prefix
        (conversion.code_ids ());
    value_slot =
      back_to_numbers ~sort:"value slot" ~prefix:Printer.value_slot_prefix
        (conversion.value_slots ());
    function_slot =
      back_to_numbers ~sort:"function slot" ~prefix:Printer.function_slot_prefix
        (conversion.function_slots ())
  }

(* ---- Reading back ------------------------------------------------------ *)

(* The text printed is read as the file was: by the compiler's parser, for the
   same compilation unit, and then normalised as the unit of the file was. *)
let read_back (conversion : conversion) ~comp_unit text :
    (Flambda_unit.t, string) result =
  let file = Filename.temp_file "flroundtrip" ".fl" in
  Misc.try_finally
    ~always:(fun () -> Misc.remove_file file)
    (fun () ->
      Out_channel.with_open_text file (fun out -> output_string out text);
      match Parse_flambda.parse_fexpr file with
      | exception exn ->
        (* The parser itself can raise, where it makes a constant of what it
           read: so it does of a naked integer outside 63 bits that stands in an
           expression (label C1-11). That is no error that it reports. *)
        Error ("the parser of fexpr fails: " ^ Printexc.to_string exn)
      | Error (Parsing_error (msg, loc)) ->
        Error
          (Format.asprintf "syntax error at line %d: %s" loc.loc_start.pos_lnum
             msg)
      | Error (Lexing_error (error, loc)) ->
        Error
          (Format.asprintf "lexing error at line %d: %a" loc.loc_start.pos_lnum
             Flambda_lex.pp_error error)
      | Ok fexpr -> (
        match Fexpr_to_flambda.conv comp_unit fexpr with
        | result -> conversion.as_read result.unit
        | exception exn ->
          Error
            ("the conversion of fexpr to Flambda fails: "
           ^ Printexc.to_string exn)))

(* ---- Where two terms differ -------------------------------------------- *)

let printed ~value_slot_kind (unit : converted) =
  match
    Printer.unit ~value_slot_kind ~return_continuation:unit.return_continuation
      ~exn_continuation:unit.exn_continuation unit.body
  with
  | fexpr -> Ok fexpr
  | exception Printer.Not_printable reason -> Error reason

(* The two terms printed with their variables and continuations named in the
   order in which they are met, so that terms that differ by bound names only
   are printed the same; and the first line that differs. *)
let first_difference ~value_slot_kind (first : converted) (second : converted) =
  let lines unit =
    match
      Printer.naming_by_order_met (fun () -> printed ~value_slot_kind unit)
    with
    | Ok fexpr -> String.split_on_char '\n' (Printer.to_string fexpr)
    | Error reason -> ["(not printable: " ^ reason ^ ")"]
  in
  let rec go line first second =
    match first, second with
    | [], [] -> None
    | a :: first, b :: second when String.equal a b ->
      go (line + 1) first second
    | a :: _, b :: _ -> Some (line, a, b)
    | a :: _, [] -> Some (line, a, "(nothing)")
    | [], b :: _ -> Some (line, "(nothing)", b)
  in
  go 1 (lines first) (lines second)

(* ---- The check --------------------------------------------------------- *)

type outcome =
  | The_same
  | Differs of (int * string * string) option
  | Not_printable of string
  | Not_read_back of string
  | Names_not_read_back of string
  | Not_converted of string
  | Not_converted_back of string

let round_trip (conversion : conversion) ~comp_unit (unit : Flambda_unit.t) :
    outcome =
  conversion.reset ();
  match conversion.convert unit with
  | Error reason -> Not_converted reason
  | Ok first -> (
    (* The kinds of the value slots are those of the first conversion: they are
       read now, before the names are forgotten. *)
    let slot_kinds =
      List.map
        (fun (_, slot) -> slot, conversion.value_slot_kind slot)
        (conversion.value_slots ())
    in
    let value_slot_kind slot = Option.join (List.assoc_opt slot slot_kinds) in
    match printed ~value_slot_kind first with
    | Error reason -> Not_printable reason
    | Ok fexpr -> (
      match read_back conversion ~comp_unit (Printer.to_string fexpr) with
      | Error reason -> Not_read_back reason
      | Ok unit_read -> (
        conversion.reset ();
        match conversion.convert unit_read with
        | Error reason -> Not_converted_back reason
        | Ok second -> (
          match rename_expr (renaming_of conversion) second.body with
          | exception Names_not_read_back reason -> Names_not_read_back reason
          | body ->
            let first = { first with body = slots_in_order first.body } in
            let second = { second with body = slots_in_order body } in
            let continuations_of_the_units =
              [ first.return_continuation, second.return_continuation;
                first.exn_continuation, second.exn_continuation ]
            in
            if
              Compare.expr_eqb [] continuations_of_the_units first.body
                second.body
            then The_same
            else Differs (first_difference ~value_slot_kind first second)))))

(* A reason, on one line. *)
let one_line reason =
  String.split_on_char '\n' reason
  |> List.map String.trim
  |> List.filter (fun line -> not (String.equal line ""))
  |> String.concat " "

let print_outcome ~filename (outcome : outcome) =
  match outcome with
  | The_same -> Format.printf "%s: round trip the same@." filename
  | Differs where -> (
    Format.printf "%s: round trip DIFFERS@." filename;
    match where with
    | Some (line, first, second) ->
      Format.printf
        "  the two terms, printed, differ first at line %d@.  converted: %s@.  \
         read back: %s@."
        line (String.trim first) (String.trim second)
    | None ->
      Format.printf
        "  the two terms are printed the same: they differ in what the printer \
         does not show@.")
  | Not_printable reason ->
    Format.printf "%s: not printable: %s@." filename (one_line reason)
  | Not_read_back reason ->
    Format.printf "%s: not read back: %s@." filename (one_line reason)
  | Names_not_read_back reason ->
    Format.printf "%s: names not read back: %s@." filename (one_line reason)
  | Not_converted reason ->
    Format.printf "%s: not converted: the model has not got %s@." filename
      (one_line reason)
  | Not_converted_back reason ->
    Format.printf
      "%s: not converted back: the model has not got %s, in the unit read \
       back@."
      filename (one_line reason)

(* [simplify] runs Simplify on the unit of the file, given as its text was
   parsed and as a term, or says why it is not to be run: on a unit that is not
   well-formed, or that is not the compiler's because its text has lost the
   arity of a call, Simplify is entitled to fail. *)
let check_file (conversion : conversion)
    ~(simplify :
       Fexpr.flambda_unit ->
       Fexpr_to_flambda.conv_result ->
       (Flambda_unit.t, string) Result.t) filename =
  match Parse_flambda.parse_fexpr filename with
  | Error e ->
    Test_utils.dump_error e;
    exit 1
  | Ok fexpr -> (
    let unit_info = Parse_flambda.make_unit_info ~filename in
    let comp_unit = Unit_info.modname unit_info in
    (* As the check of a unit sets it (flsemantics.ml, [check_file]). *)
    Compilenv.reset unit_info;
    let parsed = Fexpr_to_flambda.conv comp_unit fexpr in
    let filename = Filename.basename filename in
    let check ~filename unit =
      let outcome =
        Misc.Colours.without_colours ~f:(fun () ->
            round_trip conversion ~comp_unit unit)
      in
      print_outcome ~filename outcome;
      (* Set FLROUNDTRIP_PRINT to see the text that is read back. *)
      if Sys.getenv_opt "FLROUNDTRIP_PRINT" <> None
      then (
        conversion.reset ();
        match conversion.convert unit with
        | Error _ -> ()
        | Ok first -> (
          match printed ~value_slot_kind:conversion.value_slot_kind first with
          | Ok fexpr -> print_string (Printer.to_string fexpr)
          | Error _ -> ()))
    in
    (match conversion.as_read parsed.unit with
    | Ok unit -> check ~filename unit
    | Error why ->
      Format.printf "%s: not converted: %s@." filename (one_line why));
    let filename = filename ^ ", after Simplify" in
    match simplify fexpr parsed with
    | Ok simplified -> check ~filename simplified
    | Error why ->
      Format.printf "%s: not converted: Simplify is not run, as %s@." filename
        why)
