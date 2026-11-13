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

(* Runs the Rocq transcriptions of the reaper beside the compiler's reaper on
   one example, and compares what the two compute.

   This file has no command line of its own and does nothing when it is linked:
   flreaper_main.ml is the program, and calls [main].

   An [.fl] file is read and given to Simplify as flsemantics.ml does it (the
   same normalisation of the text, the same test that the unit is well-formed),
   and the unit that Simplify returns, which is what the compiler's reaper is
   given, is the unit of the comparison. The reaper is run in two modes, both
   without unboxing: first changing no calling convention (the flags
   [-no-reaper-change-calling-conventions] and [-no-reaper-unbox], which the
   tool sets whatever its command line says: [main]), then changing calling
   conventions, which is the last part of what is printed
   ([compare_calling_conventions]); the mode of [-reaper-preserve-direct-calls]
   and of [-reaper-local-fields] is the one that the command line gives, the
   compiler's default unless it says otherwise. Then:

   - The unit is converted to a term of the model by the conversion of
   flsemantics.ml, and the extracted traversal
   (FlambdaReaperTraversal.traverse_unit) computes the graph of the model.

   - A solution of the model's graph is computed here, by the naive Datalog
   solver [solve]. It is not trusted: the extracted checks
   (FlambdaReaperSolution.check_closed and check_field_used_exact, proved to
   imply FlambdaReaperGraphSpec.closed and field_used_exact) test it, and the
   tool fails if they do not pass.

   - The compiler's graph is read off the reaper's own traversal
   ([Traverse.run], completed as [Analysis.solve] completes it), and its
   solution off the Datalog database that points_to_analysis.ml computes from
   that graph. Both are read into the model's terms ([compiler_graph]), and the
   same extracted checks are run on them: whether the compiler's solution is
   closed under the rules of the model for the compiler's own graph. That is
   printed and nothing depends on it.

   - The two solutions are compared on the names of the unit: its variables,
   symbols and code ids, as the conversion numbered them, and the node of
   constants. The two traversals also make nodes of their own (the variables of
   calls, of returns, of the closures that a set does not bind), which do not
   correspond from one graph to the other by any name, and are not compared.
   What is compared of a name is [has_usage], [has_source], [any_usage],
   [any_source] and the fields for which [field_of_constructor_is_used] holds.
   The facts of the two graphs whose nodes are all names of the unit are
   compared too, and the differences printed, so that a difference between the
   solutions can be traced to the graphs.

   - The compiler's reaper is run on the unit, and the extracted rebuild
   (FlambdaReaperRebuild.rebuild_unit_from_tables) on the term of the model,
   with the model's graph and its solution once checked. The two terms are
   compared up to bound names, and run on the machine extracted from
   FlambdaSemanticsSpec.v beside the reaper's input ([compare_rebuild]).

   This is a test and not a proof. What is trusted: the conversion of
   flsemantics.ml (its header says what that is), and, for the comparison only,
   the reading of the compiler's graph and solution here, which must send each
   node, field and relation of the compiler to the one of the model of the same
   name ([model_node], [model_field], [compiler_graph]), the variables of the
   compiler's traversal through the walk that relates them to those of the
   conversion ([relate_variables]). What is not trusted: the solver [solve],
   which is checked; the traversal and the rebuild of the model, which are
   compared. A difference between the two solutions, the two graphs, the two
   rebuilt terms or the runs is a result and not a failure of the tool: it is
   printed, and kept in the expected file of the example. The tool fails when
   the model's solution does not pass the checks, which is a fault of [solve] or
   of the checks; the model's rebuild is then not run.

   Known differences, from the status of S7
   (middle_end/flambda2/rocq/status/S7.md) and the header of
   FlambdaReaperTraversal.v: the model has one kind of indirect call, which its
   traversal reads as a call of unknown arity, so a unit with a call of known
   arity has another graph in the model, and the tool says so; and the closures
   of a set other than the first are bound in the model by projections of the
   first, where the compiler binds them directly. From the conversion: the end
   of a region is the unit value in the model, where the compiler's traversal
   gives the primitive a use that it does not track; and a call that never
   returns is a call that returns to a continuation of the conversion's, whose
   parameters the model's rebuild may remove, so that the two rebuilt terms then
   differ (tests/semantics/arity_text_never_returns.fl). *)

(* The matches below on the compiler's types name the cases the model has and
   send every other to one answer; they are meant to stay as they are when the
   compiler gains a constructor. *)
[@@@ocaml.warning "-4"]

open Import
open Flambda2_bound_identifiers
module Graph_spec = Flambda2_rocq_semantics.FlambdaReaperGraphSpec
module Traversal = Flambda2_rocq_semantics.FlambdaReaperTraversal
module Solution = Flambda2_rocq_semantics.FlambdaReaperSolution
module Rebuild = Flambda2_rocq_semantics.FlambdaReaperRebuild

module Cc = Flambda2_rocq_semantics.FlambdaReaperCallingConventions

module Model_compare = Flambda2_rocq_semantics.FlambdaCompareSpec
module Imports = Flambda2_rocq_semantics.FlambdaImportsSpec
module Rocq_model = Flambda2_rocq_semantics.FlambdaModel
module Kinds = Flambda2_rocq_semantics.FlambdaKindModel
module Datatypes = Flambda2_rocq_library.Datatypes
module Numbers = Flambda2_rocq_library.BinNums
module Reaper = Flambda2_reaper
module PTA = Flambda2_reaper.Points_to_analysis
module GFG = Flambda2_reaper.Global_flow_graph
module Datalog = Flambda2_datalog.Datalog
module Features = Flambda2_ui.Flambda_features

(* ---- Numbers ---------------------------------------------------------- *)

let rec int_of_nat (n : Datatypes.nat) =
  match n with O -> 0 | S n -> 1 + int_of_nat n

let nat_of_int = Flsemantics.nat_of_int

let int_of_positive = Flsemantics.int_of_positive

(* ---- The names of the unit -------------------------------------------- *)

(* The names of the unit as the conversion numbered them, taken once the
   simplified unit is converted and before anything else is numbered. The
   variables are those that the conversion met: opening a binder of a term gives
   it fresh names ([Name_abstraction.pattern_match]), so the compiler's
   traversal, which opens the same binders, has other variables for them, which
   [relate_variables] relates to these. Symbols and code ids are not renamed. *)
type unit_names =
  { variables : Numbers.positive Variable.Map.t;
    symbols : Numbers.positive Symbol.Map.t;
    code_ids : Numbers.positive Code_id.Map.t;
    (* The symbol that the compiler's traversal makes for the constants of the
       unit ([Traverse.run]), which is [Node_all_constants] of the model. *)
    all_constants : Symbol.t
  }

let take_unit_names () =
  let all_constants =
    Symbol.create
      (Current_unit.get_cu_exn ())
      (Linkage_name.of_string "all_constants")
  in
  { variables = Variable.Map.of_list (Flsemantics.Variables.bindings ());
    symbols = Symbol.Map.of_list (Flsemantics.Symbols.bindings ());
    code_ids = Code_id.Map.of_list (Flsemantics.Code_ids.bindings ());
    all_constants
  }

(* The nodes of the model that stand for names of the unit, in the order of the
   numbers. *)
let unit_nodes (names : unit_names) : Graph_spec.node list =
  let numbers map = List.sort compare (List.map snd map) in
  List.map
    (fun n -> Graph_spec.Node_variable (Flsemantics.positive_of_int n))
    (numbers
       (List.map
          (fun (v, p) -> v, int_of_positive p)
          (Variable.Map.bindings names.variables)))
  @ List.map
      (fun n -> Graph_spec.Node_symbol (Flsemantics.positive_of_int n))
      (numbers
         (List.map
            (fun (s, p) -> s, int_of_positive p)
            (Symbol.Map.bindings names.symbols)))
  @ List.map
      (fun n -> Graph_spec.Node_code_id (Flsemantics.positive_of_int n))
      (numbers
         (List.map
            (fun (c, p) -> c, int_of_positive p)
            (Code_id.Map.bindings names.code_ids)))
  @ [Graph_spec.Node_all_constants]

let is_unit_node (node : Graph_spec.node) =
  match node with
  | Node_variable _ | Node_symbol _ | Node_code_id _ | Node_all_constants ->
    true
  | Node_made _ -> false

(* ---- Printing --------------------------------------------------------- *)

let print_node ppf (node : Graph_spec.node) =
  match node with
  | Node_variable v -> (
    match Flsemantics.Variables.name_of v with
    | Some var ->
      Format.fprintf ppf "%s/v%d" (Variable.name var) (int_of_positive v)
    | None -> Format.fprintf ppf "v%d" (int_of_positive v))
  | Node_symbol s -> (
    match Flsemantics.Symbols.name_of s with
    | Some symbol ->
      Format.fprintf ppf "$%s"
        (Linkage_name.to_string (Symbol.linkage_name symbol))
    | None -> Format.fprintf ppf "$s%d" (int_of_positive s))
  | Node_code_id c -> (
    match Flsemantics.Code_ids.name_of c with
    | Some code_id -> Format.fprintf ppf "code %s" (Code_id.name code_id)
    | None -> Format.fprintf ppf "code c%d" (int_of_positive c))
  | Node_all_constants -> Format.fprintf ppf "all_constants"
  | Node_made n -> Format.fprintf ppf "#%d" (int_of_positive n)

let string_of_kind (kind : Kinds.flambda_kind) =
  match kind with
  | Value -> "val"
  | Naked_number _ -> "naked"
  | Region -> "region"
  | Rec_info -> "rec_info"

let string_of_boxable_number (kind : Kinds.boxable_number) =
  match kind with
  | Boxable_naked_float32 -> "float32"
  | Boxable_naked_float -> "float"
  | Boxable_naked_int32 -> "int32"
  | Boxable_naked_int64 -> "int64"
  | Boxable_naked_nativeint -> "nativeint"
  | Boxable_naked_vec128 -> "vec128"
  | Boxable_naked_vec256 -> "vec256"
  | Boxable_naked_vec512 -> "vec512"
  | Boxable_naked_mask -> "mask"

let print_field ppf (field : Graph_spec.field) =
  match field with
  | Field_block (i, kind) ->
    Format.fprintf ppf "field %d %s" (int_of_nat i) (string_of_kind kind)
  | Field_value_slot w -> (
    match Flsemantics.Value_slots.name_of w with
    | Some slot -> Format.fprintf ppf "value slot %s" (Value_slot.name slot)
    | None -> Format.fprintf ppf "value slot w%d" (int_of_positive w))
  | Field_function_slot f -> (
    match Flsemantics.Function_slots.name_of f with
    | Some slot ->
      Format.fprintf ppf "function slot %s" (Function_slot.name slot)
    | None -> Format.fprintf ppf "function slot f%d" (int_of_positive f))
  | Field_call_witness Known_arity_code_pointer ->
    Format.fprintf ppf "call witness (known arity)"
  | Field_call_witness Unknown_arity_code_pointer ->
    Format.fprintf ppf "call witness (unknown arity)"
  | Field_is_int -> Format.fprintf ppf "is_int"
  | Field_get_tag -> Format.fprintf ppf "get_tag"
  | Field_boxed_number kind ->
    Format.fprintf ppf "boxed %s" (string_of_boxable_number kind)
  | Field_return_of_call (Normal i) ->
    Format.fprintf ppf "return %d" (int_of_nat i)
  | Field_return_of_call Exn -> Format.fprintf ppf "exn return"
  | Field_code_id_of_call_witness ->
    Format.fprintf ppf "code id of call witness"

(* A fact of a graph, of any relation, so that the facts of two graphs can be
   put side by side. *)
type fact =
  | Alias of Graph_spec.node * Graph_spec.node
  | Use of Graph_spec.node * Graph_spec.node
  | Accessor of Graph_spec.node * Graph_spec.field * Graph_spec.node
  | Constructor of Graph_spec.node * Graph_spec.field * Graph_spec.node
  | Argument of Graph_spec.node * Datatypes.nat * Graph_spec.node
  | Parameter of Graph_spec.node * Datatypes.nat * Graph_spec.node
  | Propagate of Graph_spec.node * Graph_spec.node * Graph_spec.node
  | Alias_if_any_source of Graph_spec.node * Graph_spec.node * Graph_spec.node
  | Any_usage of Graph_spec.node
  | Any_source of Graph_spec.node
  | Any_source_if_used of Graph_spec.node * Graph_spec.node
  | Zero_alloc_source of Graph_spec.node

let facts_of (f : Graph_spec.facts) : fact list =
  List.map (fun (to_, from) -> Alias (to_, from)) f.f_alias
  @ List.map (fun (to_, from) -> Use (to_, from)) f.f_use
  @ List.map
      (fun ((to_, field), base) -> Accessor (to_, field, base))
      f.f_accessor
  @ List.map
      (fun ((base, field), from) -> Constructor (base, field, from))
      f.f_constructor
  @ List.map (fun ((from, i), base) -> Argument (from, i, base)) f.f_argument
  @ List.map (fun ((base, i), to_) -> Parameter (base, i, to_)) f.f_parameter
  @ List.map
      (fun ((if_used, to_), from) -> Propagate (if_used, to_, from))
      f.f_propagate
  @ List.map
      (fun ((if_any_source, to_), from) ->
        Alias_if_any_source (if_any_source, to_, from))
      f.f_alias_if_any_source
  @ List.map (fun x -> Any_usage x) f.f_any_usage
  @ List.map (fun x -> Any_source x) f.f_any_source
  @ List.map
      (fun (if_used, x) -> Any_source_if_used (if_used, x))
      f.f_any_source_if_used
  @ List.map (fun x -> Zero_alloc_source x) f.f_zero_alloc_source

let nodes_of_fact (fact : fact) =
  match fact with
  | Alias (a, b) | Use (a, b) | Any_source_if_used (a, b) -> [a; b]
  | Accessor (a, _, b)
  | Constructor (a, _, b)
  | Argument (a, _, b)
  | Parameter (a, _, b) ->
    [a; b]
  | Propagate (a, b, c) | Alias_if_any_source (a, b, c) -> [a; b; c]
  | Any_usage a | Any_source a | Zero_alloc_source a -> [a]

(* Written as in the comments of global_flow_graph.mli: [to_ = from] for an
   alias, [to_ = base.field] for an accessor, [base = { field = from }] for a
   constructor, [base(i = from)] for an argument and [to_ = base@i] for a
   parameter. *)
let print_fact ppf (fact : fact) =
  let node = print_node in
  match fact with
  | Alias (to_, from) -> Format.fprintf ppf "alias %a = %a" node to_ node from
  | Use (to_, from) -> Format.fprintf ppf "use %a = [...] %a" node to_ node from
  | Accessor (to_, field, base) ->
    Format.fprintf ppf "accessor %a = %a.(%a)" node to_ node base print_field
      field
  | Constructor (base, field, from) ->
    Format.fprintf ppf "constructor %a = { %a = %a }" node base print_field
      field node from
  | Argument (from, i, base) ->
    Format.fprintf ppf "argument %a(%d = %a)" node base (int_of_nat i) node from
  | Parameter (base, i, to_) ->
    Format.fprintf ppf "parameter %a = %a@%d" node to_ node base (int_of_nat i)
  | Propagate (if_used, to_, from) ->
    Format.fprintf ppf "propagate %a = %a if %a is used" node to_ node from node
      if_used
  | Alias_if_any_source (if_any_source, to_, from) ->
    Format.fprintf ppf "alias_if_any_source %a = %a if %a has any source" node
      to_ node from node if_any_source
  | Any_usage x -> Format.fprintf ppf "any_usage %a" node x
  | Any_source x -> Format.fprintf ppf "any_source %a" node x
  | Any_source_if_used (if_used, x) ->
    Format.fprintf ppf "any_source %a if %a is used" node x node if_used
  | Zero_alloc_source x -> Format.fprintf ppf "zero_alloc_source %a" node x

(* The facts of a graph, and the tables of a solution, one to a line; for
   [FLREAPER_PRINT_GRAPHS]. *)
let print_graph ~title (facts : Graph_spec.facts) =
  Format.printf "  %s:@." title;
  List.iter
    (fun fact -> Format.printf "    %a@." print_fact fact)
    (facts_of facts)

let print_tables ~title (tables : Graph_spec.tables) =
  let nodes name l =
    Format.printf "    %s: %a@." name
      (Format.pp_print_list ~pp_sep:Format.pp_print_space print_node)
      l
  in
  let pairs name l =
    List.iter
      (fun (x, y) ->
        Format.printf "    %s %a %a@." name print_node x print_node y)
      l
  in
  Format.printf "  %s:@." title;
  pairs "alias" tables.t_alias;
  nodes "any_usage" tables.t_any_usage;
  nodes "any_source" tables.t_any_source;
  nodes "has_usage" tables.t_has_usage;
  nodes "has_source" tables.t_has_source;
  pairs "usages" tables.t_usages;
  pairs "sources" tables.t_sources;
  List.iter
    (fun (f, x) ->
      Format.printf "    reading_field %a %a@." print_field f print_node x)
    tables.t_reading_field;
  List.iter
    (fun (f, x) ->
      Format.printf "    escaping_field %a %a@." print_field f print_node x)
    tables.t_escaping_field;
  List.iter
    (fun (x, f) ->
      Format.printf "    field_of_constructor_is_used %a %a@." print_node x
        print_field f)
    tables.t_field_used

(* ---- Local fields ----------------------------------------------------- *)

(* [Analysis_scope.is_local_field] for the scope of the compiler's reaper
   ([Current_unit]), read of a field of the model through the names that the
   conversion gave its slots. It depends on the flag [-reaper-local-fields]. *)
let is_local (field : Graph_spec.field) =
  let local compiler_field =
    Reaper.Analysis_scope.is_local_field Current_unit compiler_field
  in
  match field with
  | Field_value_slot w -> (
    match Flsemantics.Value_slots.name_of w with
    | Some slot -> local (Field.value_slot slot)
    | None -> false)
  | Field_function_slot f -> (
    match Flsemantics.Function_slots.name_of f with
    | Some slot -> local (Field.function_slot slot)
    | None -> false)
  | Field_block _ | Field_call_witness _ | Field_is_int | Field_get_tag
  | Field_boxed_number _ | Field_return_of_call _
  | Field_code_id_of_call_witness ->
    false

(* ---- The solver ------------------------------------------------------- *)

(* A set that remembers whether it has grown, and the order in which its
   elements came. The elements are terms of the model, compared structurally. *)
type 'a set =
  { table : ('a, unit) Hashtbl.t;
    mutable elements : 'a list
  }

let new_set () = { table = Hashtbl.create 64; elements = [] }

let mem set x = Hashtbl.mem set.table x

let elements set = List.rev set.elements

(* The facts of a relation, indexed by some of their arguments. *)
let index (l : 'a list) ~(key : 'a -> 'k) ~(value : 'a -> 'v) =
  let table = Hashtbl.create 64 in
  List.iter (fun x -> Hashtbl.add table (key x) (value x)) l;
  table

(* A solution of the rules of FlambdaReaperGraphSpec.closed for the graph
   [facts], naively: each rule is a loop over the facts and the tuples that its
   premises name, which adds its conclusion, and the rules are applied until
   nothing is added.

   The rules whose premise is a guard ([any_usage x = false] or [any_source x =
   false], from [nontop_usages] and [nontop_sources] of points_to_analysis.ml)
   make the rules not monotone, and so which fixpoint is reached depends on when
   a guard is read. The rules are applied as the compiler's schedule applies
   them ([Datalog_schedule]): those that the compiler writes with [let$] are
   applied until nothing is added, then those written with [let$$], which are
   the guarded rules and some others, until nothing is added, and the two again
   in turn until neither adds anything. No [let$$] rule adds to [any_usage] or
   [any_source], so a guard is read once the [let$] rules have been saturated,
   where the compiler reads it. The rules of [field_of_constructor_is_used] are
   applied once at the end, as the compiler's [post_processing_rules] are. The
   tables are those of the solution at that point, whatever the order of their
   tuples. [check_closed] and [check_field_used_exact] are what says that this
   is a solution; nothing here is trusted. *)
let solve (f : Graph_spec.facts) : Graph_spec.tables =
  let changed = ref false in
  let add set x =
    if not (mem set x)
    then (
      Hashtbl.replace set.table x ();
      set.elements <- x :: set.elements;
      changed := true)
  in
  let alias = new_set () in
  let any_usage = new_set () in
  let any_source = new_set () in
  let has_usage = new_set () in
  let has_source = new_set () in
  let usages = new_set () in
  let sources = new_set () in
  let reading_field = new_set () in
  let escaping_field = new_set () in
  let field_used = new_set () in
  (* [usages x y]: the [y] of [x]; [sources x y]: the [y] of [x], and the [x] of
     [y]; [reading_field f x]: the [x] of [f]. *)
  let usages_of = Hashtbl.create 64 in
  let sources_of = Hashtbl.create 64 in
  let sourced_by = Hashtbl.create 64 in
  let reading_of = Hashtbl.create 64 in
  let add_usage x y =
    if not (mem usages (x, y))
    then (
      add usages (x, y);
      Hashtbl.add usages_of x y)
  in
  let add_source x y =
    if not (mem sources (x, y))
    then (
      add sources (x, y);
      Hashtbl.add sources_of x y;
      Hashtbl.add sourced_by y x)
  in
  let add_reading_field relation x =
    if not (mem reading_field (relation, x))
    then (
      add reading_field (relation, x);
      Hashtbl.add reading_of relation x)
  in
  (* [accessor to_ relation base]: the [to_] of [(base, relation)]; and so on,
     each relation indexed by its base and its field or parameter number. *)
  let accessor_at =
    index f.f_accessor
      ~key:(fun ((_, relation), base) -> base, relation)
      ~value:(fun ((to_, _), _) -> to_)
  in
  let constructor_at =
    index f.f_constructor
      ~key:(fun ((base, relation), _) -> base, relation)
      ~value:(fun (_, from) -> from)
  in
  let argument_at =
    index f.f_argument
      ~key:(fun ((_, relation), base) -> base, relation)
      ~value:(fun ((from, _), _) -> from)
  in
  let parameter_at =
    index f.f_parameter
      ~key:(fun ((base, relation), _) -> base, relation)
      ~value:(fun (_, to_) -> to_)
  in
  let not_any_usage x = not (mem any_usage x) in
  let not_any_source x = not (mem any_source x) in
  (* The inputs: [alias_input], [any_usage_input], [any_source_input] and
     [any_source_zero_alloc]. *)
  List.iter (add alias) f.f_alias;
  List.iter (add any_usage) f.f_any_usage;
  List.iter (add any_source) f.f_any_source;
  List.iter (add any_source) f.f_zero_alloc_source;
  (* The rules written with [let$] in points_to_analysis.ml, each named after
     the field of [closed] that states it. *)
  let unguarded =
    [ (* alias_propagate *)
      (fun () ->
        List.iter
          (fun ((if_used, to_), from) ->
            if mem any_usage if_used then add alias (to_, from))
          f.f_propagate);
      (* alias_if_any_source *)
      (fun () ->
        List.iter
          (fun ((if_any_source, to_), from) ->
            if mem any_source if_any_source then add alias (to_, from))
          f.f_alias_if_any_source);
      (* alias_escaping_reading *)
      (fun () ->
        List.iter
          (fun (relation, from) ->
            List.iter
              (fun to_ -> add alias (to_, from))
              (Hashtbl.find_all reading_of relation))
          (elements escaping_field));
      (* has_usage_any *)
      (fun () -> List.iter (add has_usage) (elements any_usage));
      (* has_usage_alias *)
      (fun () ->
        List.iter
          (fun (to_, from) -> if mem has_usage to_ then add has_usage from)
          (elements alias));
      (* has_source_any *)
      (fun () -> List.iter (add has_source) (elements any_source));
      (* has_source_alias *)
      (fun () ->
        List.iter
          (fun (to_, from) -> if mem has_source from then add has_source to_)
          (elements alias));
      (* any_usage_use *)
      (fun () ->
        List.iter
          (fun (to_, from) -> if mem has_usage to_ then add any_usage from)
          f.f_use);
      (* any_usage_alias *)
      (fun () ->
        List.iter
          (fun (to_, from) -> if mem any_usage to_ then add any_usage from)
          (elements alias));
      (* any_usage_constructor *)
      (fun () ->
        List.iter
          (fun ((base, relation), from) ->
            if mem any_usage base && not (is_local relation)
            then add any_usage from)
          f.f_constructor);
      (* any_usage_argument *)
      (fun () ->
        List.iter
          (fun ((from, _), base) ->
            if mem any_source base then add any_usage from)
          f.f_argument);
      (* any_source_if_used *)
      (fun () ->
        List.iter
          (fun (if_used, x) -> if mem any_usage if_used then add any_source x)
          f.f_any_source_if_used);
      (* any_source_alias *)
      (fun () ->
        List.iter
          (fun (to_, from) -> if mem any_source from then add any_source to_)
          (elements alias));
      (* any_source_parameter *)
      (fun () ->
        List.iter
          (fun ((base, _), to_) ->
            if mem any_usage base then add any_source to_)
          f.f_parameter);
      (* any_source_accessor *)
      (fun () ->
        List.iter
          (fun ((to_, relation), base) ->
            if mem any_source base && not (is_local relation)
            then add any_source to_)
          f.f_accessor);
      (* any_source_use *)
      (fun () ->
        List.iter
          (fun (to_, from) -> if mem has_source from then add any_source to_)
          f.f_use);
      (* escaping_field_rule *)
      (fun () ->
        List.iter
          (fun ((base, relation), from) ->
            if mem any_usage base && is_local relation
            then add escaping_field (relation, from))
          f.f_constructor);
      (* reading_field_rule *)
      (fun () ->
        List.iter
          (fun ((to_, relation), base) ->
            if mem any_source base && is_local relation
            then add_reading_field relation to_)
          f.f_accessor) ]
  in
  (* The rules written with [let$$]. *)
  let guarded =
    [ (* alias_constructor_usage and alias_local_constructor_usage *)
      (fun () ->
        List.iter
          (fun ((base, relation), from) ->
            if not_any_usage base || is_local relation
            then
              List.iter
                (fun base_use ->
                  List.iter
                    (fun to_ -> add alias (to_, from))
                    (Hashtbl.find_all accessor_at (base_use, relation)))
                (Hashtbl.find_all usages_of base))
          f.f_constructor);
      (* alias_parameter_usage *)
      (fun () ->
        List.iter
          (fun ((base, relation), to_) ->
            if not_any_usage base
            then
              List.iter
                (fun base_use ->
                  List.iter
                    (fun from -> add alias (to_, from))
                    (Hashtbl.find_all argument_at (base_use, relation)))
                (Hashtbl.find_all usages_of base))
          f.f_parameter);
      (* alias_accessor_source and alias_local_accessor_source *)
      (fun () ->
        List.iter
          (fun ((to_, relation), base) ->
            if not_any_source base || is_local relation
            then
              List.iter
                (fun base_source ->
                  List.iter
                    (fun from -> add alias (to_, from))
                    (Hashtbl.find_all constructor_at (base_source, relation)))
                (Hashtbl.find_all sources_of base))
          f.f_accessor);
      (* alias_argument_source *)
      (fun () ->
        List.iter
          (fun ((from, relation), base) ->
            if not_any_source base
            then
              List.iter
                (fun base_source ->
                  List.iter
                    (fun to_ -> add alias (to_, from))
                    (Hashtbl.find_all parameter_at (base_source, relation)))
                (Hashtbl.find_all sources_of base))
          f.f_argument);
      (* has_usage_accessor *)
      (fun () ->
        List.iter
          (fun ((to_, _), base) -> if mem has_usage to_ then add has_usage base)
          f.f_accessor);
      (* has_usage_argument *)
      (fun () ->
        List.iter
          (fun ((from, _), base) ->
            if mem has_source from then add has_usage base)
          f.f_argument);
      (* has_source_constructor and has_source_function_slot *)
      (fun () ->
        List.iter
          (fun ((base, relation), from) ->
            if mem has_source from || Graph_spec.is_function_slot relation
            then add has_source base)
          f.f_constructor);
      (* has_source_parameter *)
      (fun () ->
        List.iter
          (fun ((base, _), to_) ->
            if mem has_usage to_ then add has_source base)
          f.f_parameter);
      (* usages_accessor *)
      (fun () ->
        List.iter
          (fun (_, base) -> if not_any_usage base then add_usage base base)
          f.f_accessor);
      (* usages_argument *)
      (fun () ->
        List.iter
          (fun (_, base) -> if not_any_usage base then add_usage base base)
          f.f_argument);
      (* usages_alias *)
      (fun () ->
        List.iter
          (fun (to_, from) ->
            if not_any_usage to_ && not_any_usage from
            then
              List.iter
                (fun usage -> add_usage from usage)
                (Hashtbl.find_all usages_of to_))
          (elements alias));
      (* sources_constructor *)
      (fun () ->
        List.iter
          (fun ((base, _), _) ->
            if not_any_source base then add_source base base)
          f.f_constructor);
      (* sources_parameter *)
      (fun () ->
        List.iter
          (fun ((base, _), _) ->
            if not_any_source base then add_source base base)
          f.f_parameter);
      (* sources_alias *)
      (fun () ->
        List.iter
          (fun (to_, from) ->
            if not_any_source from && not_any_source to_
            then
              List.iter
                (fun source -> add_source to_ source)
                (Hashtbl.find_all sources_of from))
          (elements alias)) ]
  in
  (* Whether applying [rules] until nothing is added added anything. *)
  let saturate rules =
    let grown = ref false in
    let rec go () =
      changed := false;
      List.iter (fun rule -> rule ()) rules;
      if !changed
      then (
        grown := true;
        go ())
    in
    go ();
    !grown
  in
  let rec fixpoint () =
    ignore (saturate unguarded : bool);
    if saturate guarded then fixpoint ()
  in
  fixpoint ();
  (* The post-processing: [field_used_any_usage], [field_used_usage],
     [field_used_reading] and [field_used_local_source]. *)
  List.iter
    (fun ((base, relation), _) ->
      let used_by_accessor usage =
        List.exists (mem has_usage)
          (Hashtbl.find_all accessor_at (usage, relation))
      in
      if
        (mem any_usage base && not (is_local relation))
        || List.exists used_by_accessor (Hashtbl.find_all usages_of base)
        || mem any_usage base
           && List.exists (mem has_usage) (Hashtbl.find_all reading_of relation)
        || is_local relation && mem any_usage base
           && List.exists used_by_accessor (Hashtbl.find_all sourced_by base)
      then add field_used (base, relation))
    f.f_constructor;
  { t_alias = elements alias;
    t_any_usage = elements any_usage;
    t_any_source = elements any_source;
    t_has_usage = elements has_usage;
    t_has_source = elements has_source;
    t_usages = elements usages;
    t_sources = elements sources;
    t_reading_field = elements reading_field;
    t_escaping_field = elements escaping_field;
    t_field_used = elements field_used
  }

(* ---- The compiler's graph and solution ------------------------------- *)

(* The variables that the compiler's traversal gave the binders of the unit,
   each with the number that the conversion gave the same binder, and whether
   every binder was related.

   The traversal keeps, in the term it gives the rebuild ([Traverse.Skeleton.t]:
   a [Rev_expr.t] for the unit and one for each piece of code), every binder of
   the unit with the variables it opened it with. That term and the term of the
   model are gone through together, binder for binder, in the shape that the
   conversion gives the term of the model ([Flsemantics.convert_expr]): the
   [Let] of a region or of a [Rec_info] is dropped by the conversion and skipped
   here; a set of closures is one [Let] of the first closure followed by the
   projections of the others ([FlambdaModel.let_closures]); the parameters of a
   recursive handler are its invariant parameters followed by its own; a call
   that never returns is put under a continuation of its own; and code is found
   in the skeleton by its code id, which is not renamed. Where the two terms do
   not have the same shape the walk stops there, and the binders below are not
   related; their variables are then made nodes on the compiler's side, and the
   tool says so. *)
let relate_variables (skeleton : Reaper.Traverse.Skeleton.t)
    (input : Rocq_model.expr) : Numbers.positive Variable.Map.t * bool =
  let module Rev = Reaper.Rev_expr in
  let related = ref Variable.Map.empty in
  let complete = ref true in
  let relate x var = related := Variable.Map.add var x !related in
  let relate_all xs vars =
    if List.compare_lengths xs vars = 0
    then List.iter2 relate xs vars
    else complete := false
  in
  let dropped (binder : Rev.rev_expr_holed) =
    match binder with
    | Let
        { defining_expr =
            Named
              ( Prim (Variadic ((Begin_region _ | Begin_try_region _), _), _)
              | Rec_info _ );
          _
        } ->
      true
    | Hole | Let _ | Let_cont _ | Let_cont_rec _ -> false
  in
  (* The binders of a [Rev_expr.t], the outermost first. *)
  let binders (e : Rev.rev_expr) =
    let rec up acc (binder : Rev.rev_expr_holed) =
      match binder with
      | Hole -> acc
      | Let { parent; _ } | Let_cont { parent; _ } | Let_cont_rec { parent; _ }
        ->
        up (if dropped binder then acc else binder :: acc) parent
    in
    up [] e.holed_expr
  in
  let rec walk (e : Rocq_model.expr) (rev : Rev.rev_expr_holed list) =
    match e, rev with
    | ( Let (x, _, Closure_of_set _, body),
        Let { bound_pattern = Set_of_closures bound_vars; _ } :: rev ) -> (
      match List.map Bound_var.var bound_vars with
      | var :: vars ->
        relate x var;
        projections body vars rev
      | [] -> complete := false)
    | Let (x, _, _, body), Let { bound_pattern = Singleton bound_var; _ } :: rev
      ->
      relate x (Bound_var.var bound_var);
      walk body rev
    | Let_static (bindings, body), Let { bound_pattern = Static _; _ } :: rev ->
      List.iter binding bindings;
      walk body rev
    | ( Let_cont_non_recursive (_, Handler (params, _, handler), body),
        Let_cont { handler = rev_handler; _ } :: rev ) ->
      relate_all params (Bound_parameters.vars rev_handler.bound_parameters);
      walk handler (binders rev_handler.expr);
      walk body rev
    | Let_cont_non_recursive (_, _, (Apply _ as call)), [] ->
      (* The continuation that the conversion puts around a call that never
         returns. *)
      walk call []
    | ( Let_cont_recursive (handlers, body),
        Let_cont_rec { invariant_params; handlers = rev_handlers; _ } :: rev )
      ->
      let rev_handlers = Continuation.Lmap.bindings rev_handlers in
      if List.compare_lengths handlers rev_handlers = 0
      then
        List.iter2
          (fun (_, Rocq_model.Handler (params, _, handler))
               (_, (rev_handler : Rev.cont_handler)) ->
            relate_all params
              (Bound_parameters.vars invariant_params
              @ Bound_parameters.vars rev_handler.bound_parameters);
            walk handler (binders rev_handler.expr))
          handlers rev_handlers
      else complete := false;
      walk body rev
    | (Apply _ | Apply_cont _ | Switch _ | Invalid), [] -> ()
    | ( ( Let _ | Let_static _ | Let_cont_non_recursive _ | Let_cont_recursive _
        | Apply _ | Apply_cont _ | Switch _ | Invalid ),
        _ ) ->
      complete := false
  and projections body vars rev =
    match vars, body with
    | [], _ -> walk body rev
    | var :: vars, Rocq_model.Let (y, _, Project_function_slot _, body) ->
      relate y var;
      projections body vars rev
    | _ :: _, _ -> complete := false
  and binding (binding : Rocq_model.static_binding) =
    match binding with
    | Bind_code (id, Code (params, my_closure, _, _, _, body)) -> (
      match
        Option.bind (Flsemantics.Code_ids.name_of id) (fun code_id ->
            Code_id.Map.find_opt code_id skeleton.code)
      with
      | Some { params_and_body; _ } ->
        relate_all params (Bound_parameters.vars params_and_body.params);
        relate my_closure params_and_body.my_closure;
        walk body (binders params_and_body.body)
      | None -> complete := false)
    | Bind_deleted_code _ | Bind_symbol _ -> ()
  in
  walk input (binders skeleton.toplevel_expr);
  !related, !complete

(* The node of the model for a name of the compiler: the one that the conversion
   gave it, if it is a name of the unit ([variables] relates the variables of
   the compiler's traversal), [Node_all_constants] for [all_constants], and
   otherwise a made node, numbered here as met. *)
let model_node (names : unit_names)
    ~(variables : Numbers.positive Variable.Map.t)
    ~(made : (Code_id_or_name.t, int) Hashtbl.t) (name : Code_id_or_name.t) :
    Graph_spec.node =
  let made () =
    match Hashtbl.find_opt made name with
    | Some n -> Graph_spec.Node_made (Flsemantics.positive_of_int n)
    | None ->
      let n = Hashtbl.length made + 1 in
      Hashtbl.add made name n;
      Graph_spec.Node_made (Flsemantics.positive_of_int n)
  in
  Code_id_or_name.pattern_match name
    ~var:(fun var ->
      match Variable.Map.find_opt var variables with
      | Some v -> Graph_spec.Node_variable v
      | None -> made ())
    ~symbol:(fun symbol ->
      if Symbol.equal symbol names.all_constants
      then Graph_spec.Node_all_constants
      else
        match Symbol.Map.find_opt symbol names.symbols with
        | Some s -> Graph_spec.Node_symbol s
        | None -> made ())
    ~code_id:(fun code_id ->
      match Code_id.Map.find_opt code_id names.code_ids with
      | Some c -> Graph_spec.Node_code_id c
      | None -> made ())

(* The field of the model for a field of the compiler ([Field.view]). A slot is
   given the number that the conversion gave it, or a new one. *)
let model_field (field : Field.t) : Graph_spec.field =
  match Field.view field with
  | Block (i, kind) -> Field_block (nat_of_int i, Flsemantics.convert_kind kind)
  | Value_slot slot -> Field_value_slot (Flsemantics.Value_slots.find slot)
  | Function_slot slot ->
    Field_function_slot (Flsemantics.Function_slots.find slot)
  | Call_witness Known_arity_code_pointer ->
    Field_call_witness Known_arity_code_pointer
  | Call_witness Unknown_arity_code_pointer ->
    Field_call_witness Unknown_arity_code_pointer
  | Is_int -> Field_is_int
  | Get_tag -> Field_get_tag
  | Boxed_number kind ->
    Field_boxed_number (Flsemantics.convert_boxable_number kind)
  | Return_of_call (Normal i) -> Field_return_of_call (Normal (nat_of_int i))
  | Return_of_call Exn -> Field_return_of_call Exn
  | Code_id_of_call_witness -> Field_code_id_of_call_witness

let model_parameter (param : Reaper.Cofield.t) =
  match Reaper.Cofield.view param with Param i -> nat_of_int i

(* The tuples of a relation of a Datalog database, of one, two or three
   arguments. *)
let read1 atom db =
  Datalog.Cursor.fold
    (Datalog.Cursor.create ["x"] (fun [x] -> [atom x]))
    db ~init:[]
    ~f:(fun [x] acc -> x :: acc)

let read2 atom db =
  Datalog.Cursor.fold
    (Datalog.Cursor.create ["x"; "y"] (fun [x; y] -> [atom x y]))
    db ~init:[]
    ~f:(fun [x; y] acc -> (x, y) :: acc)

let read3 atom db =
  Datalog.Cursor.fold
    (Datalog.Cursor.create ["x"; "y"; "z"] (fun [x; y; z] -> [atom x y z]))
    db ~init:[]
    ~f:(fun [x; y; z] acc -> (x, y, z) :: acc)

type compiler_side =
  { compiler_facts : Graph_spec.facts;
    compiler_tables : Graph_spec.tables;
    (* Whether every binder of the unit was related ([relate_variables]). *)
    all_related : bool;
    (* What the decisions of calling conventions are made from: the database of
       the solution, the code deps of the traversal, and the code ids met in the
       graph, each with its node of the model. *)
    solved : Datalog.database;
    code_deps : Reaper.Traverse_acc.Code_dep.t Code_id.Map.t;
    code_ids : (Code_id.t * Graph_spec.node) list
  }

(* The compiler's graph of [unit] and its solution, read into the model's terms.
   The graph is that of [Traverse.run], completed as [Analysis.solve] completes
   it before solving (the toplevel return is given [any_usage], for the analysis
   scope [Current_unit], and the delayed facts are added), which is what
   FlambdaReaperTraversal.traverse_unit transcribes. The solution is the
   database that [Points_to_analysis.perform_analysis] computes from that graph,
   which is the one that [Analysis.solve] reads when the reaper unboxes nothing
   and changes no calling convention (the unboxing analysis then adds nothing
   that is read here). Each relation is read whole with a query, and each tuple
   is turned into one of the model ([model_node], [model_field],
   [model_parameter]); [input] is the term of the model that [unit] was
   converted to, by which the variables of the traversal are related to the
   names of the model ([relate_variables]). *)
let compiler_graph (names : unit_names) (unit : Flambda_unit.t)
    ~(input : Rocq_model.expr) ~free_names : compiler_side =
  let problem, skeleton =
    Reaper.Traverse.run unit ~final_typing_env:Without_types ~free_names
  in
  let variables, all_related = relate_variables skeleton input in
  let deps = problem.Reaper.Traverse.Problem.deps in
  GFG.add_any_usage deps problem.toplevel_return;
  Reaper.Traverse_acc.resolve_delayed_deps deps ~analysis_scope:Current_unit
    ~code_deps:problem.code_deps problem.delayed_deps;
  let graph = GFG.to_datalog deps in
  let solved = PTA.perform_analysis graph ~analysis_scope:Current_unit in
  let made = Hashtbl.create 64 in
  let code_ids = ref Code_id.Map.empty in
  let node name =
    let node = model_node names ~variables ~made name in
    Code_id_or_name.pattern_match name
      ~code_id:(fun code_id ->
        code_ids := Code_id.Map.add code_id node !code_ids)
      ~var:(fun _ -> ())
      ~symbol:(fun _ -> ());
    node
  in
  let nodes l = List.map node l in
  let pairs l = List.map (fun (x, y) -> node x, node y) l in
  let module R = GFG.Relations in
  let compiler_facts : Graph_spec.facts =
    { f_alias = pairs (read2 (fun to_ from -> R.alias ~to_ ~from) graph);
      f_use = pairs (read2 (fun to_ from -> R.use ~to_ ~from) graph);
      f_accessor =
        List.map
          (fun (to_, field, base) -> (node to_, model_field field), node base)
          (read3 (fun to_ field base -> R.accessor ~to_ field ~base) graph);
      f_constructor =
        List.map
          (fun (base, field, from) -> (node base, model_field field), node from)
          (read3 (fun base f from -> R.constructor ~base f ~from) graph);
      f_argument =
        List.map
          (fun (from, param, base) ->
            (node from, model_parameter param), node base)
          (read3 (fun from param base -> R.argument ~from param ~base) graph);
      f_parameter =
        List.map
          (fun (base, param, to_) ->
            (node base, model_parameter param), node to_)
          (read3 (fun base param to_ -> R.parameter ~base param ~to_) graph);
      f_propagate =
        List.map
          (fun (if_used, to_, from) -> (node if_used, node to_), node from)
          (read3
             (fun if_used to_ from -> R.propagate ~if_used ~to_ ~from)
             graph);
      f_alias_if_any_source =
        List.map
          (fun (if_any_source, to_, from) ->
            (node if_any_source, node to_), node from)
          (read3
             (fun if_any_source to_ from ->
               R.alias_if_any_source ~if_any_source ~to_ ~from)
             graph);
      f_any_usage = nodes (read1 R.any_usage graph);
      f_any_source = nodes (read1 R.any_source graph);
      f_any_source_if_used =
        pairs (read2 (fun if_used x -> R.any_source_if_used ~if_used x) graph);
      f_zero_alloc_source = nodes (read1 R.zero_alloc_source graph)
    }
  in
  let fields_and_nodes l = List.map (fun (f, x) -> model_field f, node x) l in
  let compiler_tables : Graph_spec.tables =
    { t_alias = pairs (read2 (fun to_ from -> R.alias ~to_ ~from) solved);
      t_any_usage = nodes (read1 R.any_usage solved);
      t_any_source = nodes (read1 R.any_source solved);
      t_has_usage = nodes (read1 PTA.Relations.has_usage solved);
      t_has_source = nodes (read1 PTA.Relations.has_source solved);
      t_usages = pairs (read2 PTA.Relations.usages solved);
      t_sources = pairs (read2 PTA.Relations.sources solved);
      t_reading_field =
        fields_and_nodes (read2 PTA.Relations.reading_field solved);
      t_escaping_field =
        fields_and_nodes (read2 PTA.Relations.escaping_field solved);
      t_field_used =
        List.map
          (fun (x, f) -> node x, model_field f)
          (read2 PTA.Relations.field_of_constructor_is_used solved)
    }
  in
  { compiler_facts;
    compiler_tables;
    all_related;
    solved;
    code_deps = problem.code_deps;
    code_ids = Code_id.Map.bindings !code_ids
  }

(* ---- The comparisons -------------------------------------------------- *)

let dedup l = List.sort_uniq compare l

(* The facts of a graph whose nodes are all names of the unit. *)
let facts_on_unit_names facts =
  dedup
    (List.filter
       (fun fact -> List.for_all is_unit_node (nodes_of_fact fact))
       (facts_of facts))

let compare_graphs ~(model : Graph_spec.facts) ~(compiler : Graph_spec.facts) =
  let model = facts_on_unit_names model in
  let compiler = facts_on_unit_names compiler in
  let only_in l other = List.filter (fun fact -> not (List.mem fact other)) l in
  match only_in model compiler, only_in compiler model with
  | [], [] ->
    Format.printf
      "  the facts of the two graphs on the names of the unit: the same (%d)@."
      (List.length model)
  | only_model, only_compiler ->
    Format.printf
      "  the facts of the two graphs on the names of the unit: %d in the \
       model, %d in the compiler, of which:@."
      (List.length model) (List.length compiler);
    List.iter
      (fun fact -> Format.printf "    only in the model: %a@." print_fact fact)
      only_model;
    List.iter
      (fun fact ->
        Format.printf "    only in the compiler: %a@." print_fact fact)
      only_compiler

(* What is compared of a name of the unit in a solution. *)
type of_a_name =
  { has_usage : bool;
    has_source : bool;
    any_usage : bool;
    any_source : bool;
    fields_used : Graph_spec.field list
  }

let of_a_name (tables : Graph_spec.tables) node =
  { has_usage = Graph_spec.mem_node node tables.t_has_usage;
    has_source = Graph_spec.mem_node node tables.t_has_source;
    any_usage = Graph_spec.mem_node node tables.t_any_usage;
    any_source = Graph_spec.mem_node node tables.t_any_source;
    fields_used =
      dedup
        (List.filter_map
           (fun (x, f) -> if x = node then Some f else None)
           tables.t_field_used)
  }

let compare_solutions ~(names : Graph_spec.node list)
    ~(model : Graph_spec.tables) ~(compiler : Graph_spec.tables) =
  let differences =
    List.filter_map
      (fun node ->
        let m = of_a_name model node in
        let c = of_a_name compiler node in
        if m = c then None else Some (node, m, c))
      names
  in
  let yes_no b = if b then "yes" else "no" in
  match differences with
  | [] ->
    Format.printf "  the two solutions on the %d names of the unit: the same@."
      (List.length names)
  | _ :: _ ->
    Format.printf
      "  the two solutions differ on %d of the %d names of the unit:@."
      (List.length differences) (List.length names);
    List.iter
      (fun (node, m, c) ->
        let relation name m c =
          if m <> c
          then
            Format.printf "    %a: %s: model %s, compiler %s@." print_node node
              name (yes_no m) (yes_no c)
        in
        relation "has_usage" m.has_usage c.has_usage;
        relation "has_source" m.has_source c.has_source;
        relation "any_usage" m.any_usage c.any_usage;
        relation "any_source" m.any_source c.any_source;
        let only_in l other =
          List.filter (fun field -> not (List.mem field other)) l
        in
        List.iter
          (fun field ->
            Format.printf
              "    %a: field_of_constructor_is_used %a: model yes, compiler \
               no@."
              print_node node print_field field)
          (only_in m.fields_used c.fields_used);
        List.iter
          (fun field ->
            Format.printf
              "    %a: field_of_constructor_is_used %a: model no, compiler \
               yes@."
              print_node node print_field field)
          (only_in c.fields_used m.fields_used))
      differences

(* The number of distinct facts of a graph: the traversal of the model may list
   a fact twice, which means nothing (FlambdaReaperTraversal.v). *)
let size (facts : Graph_spec.facts) = List.length (dedup (facts_of facts))

let passes ok = if ok then "passes" else "FAILS"

(* ---- The rebuild ------------------------------------------------------ *)

(* The mode of [-reaper-preserve-direct-calls], as the traversal of the model
   takes it. *)
let preserve_direct_calls () : Traversal.reaper_preserve_direct_calls =
  match Features.reaper_preserve_direct_calls () with
  | Never -> Never
  | Always -> Always
  | Zero_alloc -> Zero_alloc
  | Auto -> Auto

(* The processor time that each part of the comparison of a unit takes, printed
   on the standard error where FLREAPER_TIMES is set, so that the expected files
   do not depend on it. *)
let timed name f =
  match Sys.getenv_opt "FLREAPER_TIMES" with
  | None | Some "" -> f ()
  | Some _ ->
    let start = Sys.time () in
    let result = f () in
    Printf.eprintf "flreaper: %s: %.2fs\n%!" name (Sys.time () -. start);
    result

(* The extracted rebuild can take a time exponential in the number of groups of
   arguments of a call of unknown arity:
   [FlambdaReaperRebuild.get_field_sources] returns its set of sources as a
   predicate made of the one it is given, so that [arguments_used_by_call], for
   a call of [k] groups, tests a predicate that goes through the constructor
   facts [2k - 1] times nested, for each parameter fact. So the rebuild of the
   model is not run where that estimate, for the call of the most groups, is
   above [rebuild_budget ()] tests: FLREAPER_REBUILD_BUDGET, or a hundred
   million, which is some ten seconds (where FLREAPER_REBUILD_ALWAYS is set, it
   is run whatever the estimate). The estimate depends on the term and the graph
   only, so what is printed does not depend on the machine. The rebuild would
   not need this if the sets of sources were lists of nodes, computed once,
   rather than predicates. *)
let rebuild_budget () =
  match
    Option.bind (Sys.getenv_opt "FLREAPER_REBUILD_BUDGET") float_of_string_opt
  with
  | Some budget -> budget
  | None -> 1e8

let rec max_groups_of_unknown_calls (e : Rocq_model.expr) =
  let in_handler (Rocq_model.Handler (_, _, handler)) =
    max_groups_of_unknown_calls handler
  in
  let in_binding (binding : Rocq_model.static_binding) =
    match binding with
    | Bind_code (_, Code (_, _, _, _, _, body)) ->
      max_groups_of_unknown_calls body
    | Bind_deleted_code _ | Bind_symbol _ -> 0
  in
  let max_of l = List.fold_left max 0 l in
  match e with
  | Let (_, _, _, body) -> max_groups_of_unknown_calls body
  | Let_static (bindings, body) ->
    max
      (max_of (List.map in_binding bindings))
      (max_groups_of_unknown_calls body)
  | Let_cont_non_recursive (_, handler, body) ->
    max (in_handler handler) (max_groups_of_unknown_calls body)
  | Let_cont_recursive (handlers, body) ->
    max
      (max_of (List.map (fun (_, handler) -> in_handler handler) handlers))
      (max_groups_of_unknown_calls body)
  | Apply { ap_call_kind = Indirect; ap_args_arity; _ } ->
    List.length ap_args_arity
  | Apply { ap_call_kind = Direct _; _ } | Apply_cont _ | Switch _ | Invalid ->
    0

(* The estimate above, and whether it is within the budget. *)
let rebuild_estimate (facts : Graph_spec.facts) (e : Rocq_model.expr) =
  let groups = max_groups_of_unknown_calls e in
  let constructors = float_of_int (max 1 (List.length facts.f_constructor)) in
  let parameters = float_of_int (max 1 (List.length facts.f_parameter)) in
  let estimate =
    if groups <= 1
    then 0.
    else parameters *. (constructors ** float_of_int ((2 * groups) - 1))
  in
  ( groups,
    estimate,
    estimate <= rebuild_budget ()
    || Option.is_some (Sys.getenv_opt "FLREAPER_REBUILD_ALWAYS") )

let within_budget name ~indent ~facts ~(input : Flsemantics.converted) f =
  let groups, estimate, allowed = rebuild_estimate facts input.body in
  if allowed
  then Some (timed name f)
  else (
    Format.printf
      "%sthe model's rebuild: NOT RUN, as a call of unknown arity has %d \
       groups of arguments, which makes it take some %.0e tests \
       (FLREAPER_REBUILD_ALWAYS runs it)@."
      indent groups estimate;
    None)

(* A solution whose relations are those of [tables], as
   [FlambdaReaperGraphSpec.solution_of_tables] gives them, answered from hash
   tables instead of by going through the lists, which makes the extracted
   rebuild quadratic in the size of the tables. [solution_of_tables] tests
   membership with [node_eqb] and [field_eqb], which decide the structural
   equality of these values, as [Hashtbl.mem] does. Trusted, for speed only. *)
let solution_of_tables (t : Graph_spec.tables) : Graph_spec.solution =
  let set l =
    let table = Hashtbl.create ((2 * List.length l) + 1) in
    List.iter (fun x -> Hashtbl.replace table x ()) l;
    Hashtbl.mem table
  in
  let alias = set t.t_alias in
  let usages = set t.t_usages in
  let sources = set t.t_sources in
  let reading_field = set t.t_reading_field in
  let escaping_field = set t.t_escaping_field in
  let field_used = set t.t_field_used in
  { s_alias = (fun to_ from -> alias (to_, from));
    s_any_usage = set t.t_any_usage;
    s_any_source = set t.t_any_source;
    s_has_usage = set t.t_has_usage;
    s_has_source = set t.t_has_source;
    s_usages = (fun x y -> usages (x, y));
    s_sources = (fun x y -> sources (x, y));
    s_reading_field = (fun f x -> reading_field (f, x));
    s_escaping_field = (fun f x -> escaping_field (f, x));
    s_field_used = (fun x f -> field_used (x, f))
  }

let print_terms () =
  match Sys.getenv_opt "FLREAPER_PRINT_TERMS" with
  | None | Some "" -> false
  | Some _ -> true

(* A term of the model, its variables and continuations numbered from 1 as they
   are met, so that two terms that are the same up to bound names print alike
   ([Flsemantics.names_shown]). *)
let print_term ~title (e : Rocq_model.expr) =
  Flsemantics.names_shown := true;
  Flsemantics.forget_shown_names ();
  Format.printf "  %s:@.    @[<v>%a@]@." title Flsemantics.print_expr e;
  Flsemantics.names_shown := false

(* What a run of the second term comes to against one of the first, as
   flsemantics.ml says it of the runs before and after Simplify. *)
let verdict (first : Flsemantics.outcome) (second : Flsemantics.outcome) =
  match first, second with
  | Undefined, _ -> "anything goes, as the run of the input is undefined"
  | Still_running, _ | _, Still_running -> "not decided"
  | _, _ -> if first = second then "the same" else "DIFFERENT"

(* The rebuild of the model ([FlambdaReaperRebuild.rebuild_unit_from_tables],
   extracted) of [input], the term of the model that the reaper is given, with
   the graph of the model and its solution, which have passed the checks; beside
   it, [compiler_output], the unit that the compiler's reaper returned,
   converted to the model. The two terms are compared up to bound names
   ([FlambdaCompareSpec.expr_eqb], the comparison of the validation of
   flsemantics.ml), and the three are run on the machine, the outcome of each
   rebuild being compared with that of the input as flsemantics.ml compares the
   runs before and after Simplify.

   The compiler's output is converted with the numbering that [input] was
   converted with (the conversion is not reset between the two), so that a name
   that the compiler's rebuild keeps has the same number on the two sides. That
   is enough for the comparison: the reaper, changing no calling convention,
   gives no code a new code id, and the symbols are those of the unit. Should
   the two units have different return or exception continuations, the second is
   given the number of the first, as [Flsemantics.convert_both] does. A
   difference of the two terms, or of a run, is printed and is not a failure of
   the tool. Setting FLREAPER_PRINT_TERMS prints the three terms. *)
let compare_rebuild ~indent ~(input : Flsemantics.converted)
    ~(input_unit : Flambda_unit.t) ~(rebuilt_body : Rocq_model.expr option)
    ~(compiler_output : Flambda_unit.t) =
  let rebuilt =
    Option.map
      (fun body : Flsemantics.converted -> { input with body })
      rebuilt_body
  in
  Flsemantics.Continuations.alias
    (Flambda_unit.return_continuation compiler_output)
    ~same_as:(Flambda_unit.return_continuation input_unit);
  Flsemantics.Continuations.alias
    (Flambda_unit.exn_continuation compiler_output)
    ~same_as:(Flambda_unit.exn_continuation input_unit);
  let output =
    match Flsemantics.convert_unit compiler_output with
    | exception Flsemantics.Unsupported what ->
      Format.printf
        "%sthe compiler's output: not converted, the model has not got %t@."
        indent what;
      None
    | output -> Some output
  in
  (match rebuilt, output with
  | None, _ | _, None -> ()
  | Some rebuilt, Some output ->
    Format.printf "%sthe model's rebuild and the compiler's output: %s@." indent
      (if
         timed "the comparison of the terms" (fun () ->
             Model_compare.expr_eqb [] [] rebuilt.body output.body)
       then "the same up to bound names"
       else "DIFFERENT terms"));
  if print_terms ()
  then (
    print_term ~title:"the reaper's input" input.body;
    Option.iter
      (fun (rebuilt : Flsemantics.converted) ->
        print_term ~title:"the model's rebuild" rebuilt.body)
      rebuilt;
    Option.iter
      (fun (output : Flsemantics.converted) ->
        print_term ~title:"the compiler's output" output.body)
      output);
  let statics =
    Imports.imports_statics
      (Flsemantics.imports ~bound:[] ~unmentioned:[] ~hidden:[])
  in
  let run ~title unit =
    let outcome, steps =
      timed ("the run of " ^ title) (fun () -> Flsemantics.run ~statics unit)
    in
    Format.printf "%s%s (%d steps): %a@." indent title steps
      Flsemantics.print_outcome outcome;
    outcome
  in
  let outcome_input = run ~title:"the reaper's input" input in
  Option.iter
    (fun rebuilt ->
      let outcome_rebuilt = run ~title:"the model's rebuild" rebuilt in
      Format.printf "%sthe model's rebuild against the input: %s@." indent
        (verdict outcome_input outcome_rebuilt))
    rebuilt;
  Option.iter
    (fun output ->
      let outcome_output = run ~title:"the compiler's output" output in
      Format.printf "%sthe compiler's output against the input: %s@." indent
        (verdict outcome_input outcome_output))
    output

(* ---- Changes of calling convention ------------------------------------ *)

let print_decision ppf (decision : Cc.param_decision) =
  match decision with
  | Keep kind -> Format.fprintf ppf "keep %s" (string_of_kind kind)
  | Delete -> Format.fprintf ppf "delete"

let print_change ppf (change : Cc.calling_convention_change option) =
  let decisions =
    Format.pp_print_list
      ~pp_sep:(fun ppf () -> Format.fprintf ppf ", ")
      print_decision
  in
  match change with
  | None -> Format.fprintf ppf "an unboxing, which the model has not got"
  | Some Not_changing_calling_convention -> Format.fprintf ppf "no change"
  | Some (Changing_calling_convention (params, results)) ->
    Format.fprintf ppf "parameters [%a], results [%a]" decisions params
      decisions results

(* A decision of the compiler ([Unboxing_analysis.param_decision]) as one of the
   model, which has no [Unbox] and no subkind. *)
let model_decision (decision : Reaper.Unboxing_analysis.param_decision) :
    Cc.param_decision option =
  match decision with
  | Keep (_, kind) ->
    Some
      (Keep
         (Flsemantics.convert_kind
            (Flambda2_kinds.Flambda_kind.With_subkind.kind kind)))
  | Delete -> Some Delete
  | Unbox _ -> None

let model_change (change : Reaper.Unboxing_analysis.calling_convention_change) :
    Cc.calling_convention_change option =
  match change with
  | Not_changing_calling_convention -> Some Not_changing_calling_convention
  | Changing_calling_convention { params_decisions; return_decisions; _ } -> (
    let all decisions =
      List.fold_right
        (fun decision decisions ->
          match model_decision decision, decisions with
          | Some decision, Some decisions -> Some (decision :: decisions)
          | None, _ | _, None -> None)
        decisions (Some [])
    in
    match all params_decisions, all return_decisions with
    | Some params, Some results ->
      Some (Changing_calling_convention (params, results))
    | None, _ | _, None -> None)

(* The reaper with changes of calling convention and without unboxing
   ([-reaper-change-calling-conventions -no-reaper-unbox]), on the same unit,
   the same graphs and the same solutions as above: the analysis of
   unboxing_analysis.ml adds to the solution only the table of the code whose
   calling convention cannot change, which the traversal does not read.

   The model's table is the least ([derive_cannot_change]), and must pass
   [check_cc_closed], or the tool fails. The compiler's is read off its
   decisions, as the transcription's header says: the table is not exported, but
   [Unboxing_analysis.is_changing_calling_convention] is. A code id of the graph
   is in it unless the traversal has a code dep for it and its calling
   convention changes ([get_calling_convention_change] is a fatal error on code
   of the unit that has none, deleted code). Both tables are checked with
   [check_cc_closed], and compared on the code ids of the unit; the decisions of
   the model ([unit_calling_convention_changes]) are compared with those of
   [compute_code_changes], a code id of the unit that has no code dep having
   none; and the model's rebuild ([rebuild_unit_cc_from_tables]) is compared
   with what [Reaper.run] returns with the two flags, as [compare_rebuild] does.
   The model reads every indirect call as one of unknown arity, which keeps the
   code it may call from changing ([cc_code_id_unknown_arity]), where the
   compiler's call of known arity does not; that difference is expected. *)
let compare_calling_conventions ~(input : Flsemantics.converted)
    ~(input_unit : Flambda_unit.t) ~(unit_code_ids : Graph_spec.node list)
    ~(model_facts : Graph_spec.facts) ~(model_tables : Graph_spec.tables)
    ~(compiler : compiler_side) ~cmx_loader
    (simplified : Simplify.simplify_result) : bool =
  let indent = "    " in
  Oxcaml_flags.Flambda2.reaper_change_calling_conventions
    := Oxcaml_flags.Set true;
  Oxcaml_flags.Flambda2.reaper_unbox := Oxcaml_flags.Set false;
  Format.printf "  with changes of calling convention:@.";
  let flag = preserve_direct_calls () in
  let interface = Flsemantics.interface input in
  let model_table = Cc.derive_cannot_change model_facts model_tables in
  let model_closed = Cc.check_cc_closed model_facts model_tables model_table in
  let ua =
    Reaper.Unboxing_analysis.perform_analysis compiler.solved
      ~analysis_scope:Current_unit
  in
  let changes =
    Reaper.Unboxing_analysis.compute_code_changes ua
      ~analysis_scope:Current_unit
      ~rewrite_kind_with_subkind:(fun _ kind -> kind)
      ~rewrite_result_types:(fun ~my_closure:_ ~params:_ ~results:_ _ ->
        Flambda2_lattices.Or_unknown_or_bottom.Unknown)
      ~code_deps:compiler.code_deps
  in
  let compiler_change code_id =
    if Code_id.Map.mem code_id compiler.code_deps
    then
      Reaper.Unboxing_analysis.get_calling_convention_change changes
        ~analysis_scope:Current_unit code_id
    else Reaper.Unboxing_analysis.Not_changing_calling_convention
  in
  let compiler_table =
    List.filter_map
      (fun (code_id, node) ->
        match compiler_change code_id with
        | Not_changing_calling_convention -> Some node
        | Changing_calling_convention _ -> None)
      compiler.code_ids
  in
  let compiler_closed =
    Cc.check_cc_closed compiler.compiler_facts compiler.compiler_tables
      compiler_table
  in
  Format.printf
    "%sthe table of the code that keeps its calling convention: the model's %s \
     the check, the compiler's, with its graph, %s it@."
    indent (passes model_closed) (passes compiler_closed);
  if model_closed
  then (
    let can_change table node =
      if Graph_spec.mem_node node table then "keeps" else "may change"
    in
    (match
       List.filter
         (fun node ->
           Graph_spec.mem_node node model_table
           <> Graph_spec.mem_node node compiler_table)
         unit_code_ids
     with
    | [] ->
      Format.printf
        "%sthe two tables on the %d code ids of the unit: the same@." indent
        (List.length unit_code_ids)
    | differences ->
      List.iter
        (fun node ->
          Format.printf
            "%s%a: model %s its calling convention, compiler %s it@." indent
            print_node node
            (can_change model_table node)
            (can_change compiler_table node))
        differences);
    let deps = Cc.unit_code_deps flag interface input.body in
    let model_decisions =
      Cc.unit_calling_convention_changes model_tables model_table deps
        input.body
    in
    let decisions =
      List.map
        (fun (id, model) ->
          let compiler =
            match Flsemantics.Code_ids.name_of id with
            | Some code_id -> model_change (compiler_change code_id)
            | None -> Some Cc.Not_changing_calling_convention
          in
          id, Some model, compiler)
        model_decisions
    in
    let changing =
      List.filter
        (fun (_, model, _) ->
          match model with
          | Some (Cc.Changing_calling_convention _) -> true
          | Some Cc.Not_changing_calling_convention | None -> false)
        decisions
    in
    (match
       List.filter (fun (_, model, compiler) -> model <> compiler) decisions
     with
    | [] ->
      Format.printf
        "%sthe decisions on the %d code ids of the unit: the same, %d \
         changing@."
        indent (List.length decisions) (List.length changing)
    | differences ->
      List.iter
        (fun (id, model, compiler) ->
          Format.printf "%s%a: model %a; compiler %a@." indent print_node
            (Graph_spec.Node_code_id id) print_change model print_change
            compiler)
        differences);
    List.iter
      (fun (id, model, _) ->
        Format.printf "%s%a: %a@." indent print_node
          (Graph_spec.Node_code_id id) print_change model)
      changing;
    let preserve =
      match Features.reaper_preserve_direct_calls () with
      | Auto | Always -> true
      | Never | Zero_alloc -> false
    in
    let rebuilt_body =
      within_budget "the model's rebuild" ~indent ~facts:model_facts ~input
        (fun () ->
          let cannot_change =
            let table = Hashtbl.create 16 in
            List.iter (fun node -> Hashtbl.replace table node ()) model_table;
            Hashtbl.mem table
          in
          Cc.rebuild_unit_cc model_facts
            (solution_of_tables model_tables)
            cannot_change deps preserve input.body)
    in
    let reaped =
      Reaper.Reaper.run ~machine_width:Sixty_four ~cmx_loader
        ~all_code:simplified.all_code
        ~final_typing_env:(With_types simplified.final_typing_env)
        ~free_names:simplified.free_names simplified.unit
    in
    compare_rebuild ~indent ~input ~input_unit ~rebuilt_body
      ~compiler_output:reaped.unit);
  model_closed

(* ---- One example ------------------------------------------------------ *)

let print_graphs () =
  match Sys.getenv_opt "FLREAPER_PRINT_GRAPHS" with
  | None | Some "" -> false
  | Some _ -> true

(* The comparison on the unit that Simplify returned. Whether it found nothing
   that makes the tool fail. *)
let check ~filename ~cmx_loader (simplified : Simplify.simplify_result) : bool =
  let unit = simplified.unit in
  Flsemantics.reset_conversion ();
  match Flsemantics.convert_unit unit with
  | exception Flsemantics.Unsupported what ->
    Format.printf "%s: not checked: the model has not got %t@." filename what;
    true
  | input -> (
    let names = take_unit_names () in
    let known_arity_call = !Flsemantics.known_arity_call_met in
    let model_facts =
      timed "the traversal of the model" (fun () ->
          Traversal.traverse_unit (preserve_direct_calls ())
            (Flsemantics.interface input)
            input.body)
    in
    let model_tables = timed "the solver" (fun () -> solve model_facts) in
    let model_closed, model_exact =
      timed "the checks of the model's solution" (fun () ->
          ( Solution.check_closed is_local model_facts model_tables,
            Solution.check_field_used_exact is_local model_facts model_tables ))
    in
    match
      timed "the compiler's graph and solution" (fun () ->
          compiler_graph names unit ~input:input.body
            ~free_names:simplified.free_names)
    with
    | exception ((Misc.Fatal_error | Assert_failure _) as e) ->
      (* The compiler's traversal is entitled to stop on a term that it was not
         made for (one that Simplify has made of a text that was already its
         output, for instance); that is said, and nothing is compared. *)
      Format.printf
        "%s: not compared: the compiler's reaper stopped on the unit (%s)@."
        filename (Printexc.to_string e);
      model_closed && model_exact
    | { compiler_facts; compiler_tables; all_related; _ } as compiler ->
      let compiler_closed, compiler_exact =
        timed "the checks of the compiler's solution" (fun () ->
            ( Solution.check_closed is_local compiler_facts compiler_tables,
              Solution.check_field_used_exact is_local compiler_facts
                compiler_tables ))
      in
      Format.printf "%s:@." filename;
      if known_arity_call
      then
        Format.printf
          "  the unit has a call of known arity, which the model reads as one \
           of unknown arity: the two graphs differ there@.";
      if not all_related
      then
        Format.printf
          "  NOT ALL the variables of the compiler's traversal could be \
           related to those of the model: the term of the model and the \
           compiler's skeleton differ in shape@.";
      Format.printf "  the graph: %d facts in the model, %d in the compiler@."
        (size model_facts) (size compiler_facts);
      Format.printf
        "  the model's solution: %s the check of closure, %s the check of \
         field_of_constructor_is_used@."
        (passes model_closed) (passes model_exact);
      Format.printf
        "  the compiler's solution, with its graph: %s the check of closure, \
         %s the check of field_of_constructor_is_used@."
        (passes compiler_closed) (passes compiler_exact);
      compare_graphs ~model:model_facts ~compiler:compiler_facts;
      compare_solutions ~names:(unit_nodes names) ~model:model_tables
        ~compiler:compiler_tables;
      if print_graphs ()
      then (
        print_graph ~title:"the graph of the model" model_facts;
        print_tables ~title:"the model's solution" model_tables;
        print_graph ~title:"the graph of the compiler" compiler_facts;
        print_tables ~title:"the compiler's solution" compiler_tables);
      let checked = model_closed && model_exact in
      if not checked
      then false
      else
        let reaped =
          Reaper.Reaper.run ~machine_width:Sixty_four ~cmx_loader
            ~all_code:simplified.all_code
            ~final_typing_env:(With_types simplified.final_typing_env)
            ~free_names:simplified.free_names unit
        in
        let preserve =
          match Features.reaper_preserve_direct_calls () with
          | Auto | Always -> true
          | Never | Zero_alloc -> false
        in
        timed "the rebuilds and the runs" (fun () ->
            compare_rebuild ~indent:"  " ~input ~input_unit:unit
              ~rebuilt_body:
                (within_budget "the model's rebuild" ~indent:"  "
                   ~facts:model_facts ~input (fun () ->
                     Rebuild.rebuild_unit model_facts
                       (solution_of_tables model_tables)
                       preserve input.body))
              ~compiler_output:reaped.unit);
        timed "the changes of calling convention" (fun () ->
            compare_calling_conventions ~input ~input_unit:unit
              ~unit_code_ids:
                (List.filter
                   (fun (node : Graph_spec.node) ->
                     match node with
                     | Node_code_id _ -> true
                     | Node_variable _ | Node_symbol _ | Node_all_constants
                     | Node_made _ ->
                       false)
                   (unit_nodes names))
              ~model_facts ~model_tables ~compiler ~cmx_loader simplified))

(* The example in [filename], read and given to Simplify as flsemantics.ml does
   it ([Flsemantics.check_file]). Whether nothing made the tool fail. *)
let check_file filename : bool =
  match Parse_flambda.parse_fexpr filename with
  | Error e ->
    Test_utils.dump_error e;
    false
  | Ok text -> (
    let unit_info = Parse_flambda.make_unit_info ~filename in
    let comp_unit = Unit_info.modname unit_info in
    Compilenv.reset unit_info;
    let fl2 = Fexpr_to_flambda.conv comp_unit text in
    let filename_shown = Filename.basename filename in
    if Flsemantics.result_arity_lost text
    then (
      Format.printf "%s: not checked: %s@." filename_shown
        Flsemantics.result_arity_lost_message;
      true)
    else
      match Flsemantics.as_read_from_text fl2.unit with
      | Error why ->
        Format.printf "%s: not checked: %s@." filename_shown why;
        true
      | Ok before ->
        if not (Flsemantics.well_formed_before ~filename:filename_shown before)
        then true
        else
          let cmx_loader =
            Flambda_cmx.create_loader ~get_module_info:Flambda2.get_module_info
          in
          let simplified =
            Simplify.run ~cmx_loader ~machine_width:Sixty_four ~round:0
              ~code_slot_offsets:fl2.code_slot_offsets before
          in
          check ~filename:filename_shown ~cmx_loader simplified)

(* The command line: the options of the compiler, and one [.fl] file. It exits 1
   where the model's solution fails the checks. *)
let main () =
  let module Options =
    Oxcaml_args.Make_optcomp_options (Oxcaml_args.Default.Optmain)
  in
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
  Compenv.parse_arguments (ref Sys.argv) defer_file "flreaper";
  Location.read_clflags_from_env ();
  (* The reaper without changes of calling convention and without unboxing,
     whatever the command line said, for the first comparison;
     [compare_calling_conventions] sets the first flag for the second. *)
  Oxcaml_flags.Flambda2.reaper_change_calling_conventions
    := Oxcaml_flags.Set false;
  Oxcaml_flags.Flambda2.reaper_unbox := Oxcaml_flags.Set false;
  Flsemantics.init_load_path ();
  (* One unit to a process, as for flsemantics.ml. *)
  match !files with
  | [] -> ()
  | [file] -> if not (check_file file) then exit 1
  | _ :: _ :: _ ->
    Misc.fatal_error "flreaper checks one .fl file: run it once for each file"
