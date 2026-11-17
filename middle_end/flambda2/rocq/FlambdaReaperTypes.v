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

(* The rewriting of the types that the reaper exports, transcribed: the
   types of the symbols of the unit, which the .cmx file carries, and the
   result types of code, given the flow graph of the unit and the
   solution of the reaper's rules.

   This file transcribes middle_end/flambda2/reaper/types_rewriter.ml
   (the metadata, [follow_field], [join_metadata], [metadata_for_name],
   the module [Rewriter], [rewrite_typing_env]), the traversal that the
   rewriter is given to ([Traversals.Make], in
   middle_end/flambda2/types/traversals.ml: [rewrite_arbitrary_type],
   [rewrite], the rewriting of the heads, [destructure_expanded_head],
   [get_canonical_with], [rewrite_in_depth] and [rewrite]), with
   [rewrite_result_types] and the part of
   [rewrite_env_extension_with_extra_variables] that it calls, and the
   queries of points_to_analysis.ml that the rewriter calls. With the
   unboxing of FlambdaReaperUnboxing.v, it transcribes the result types
   of the parameters and results that are unboxed and of the fields of an
   unboxed [my_closure] ([rewrite_result_types_u]): the patterns that
   [Rewriter.patterns_for_unboxed_fields] makes of them, with no function
   slot ([classify_field_map], [unboxed_leaves]), and their
   [fold_destructuring] with [destructure_expanded_head] at the accessors
   of the patterns ([destructure_field]). It needs no review: what the
   rewritten types mean is stated in FlambdaReaperTypesSpec.v and proved
   in FlambdaReaperTypesProofs.v. It is extracted ([rewrite_typing_env],
   [rewrite_symbol_type], [rewrite_result_types],
   [rewrite_result_type]); what is of the unboxing is not.

   The inputs. The facts of the graph and the tables of the solution
   (FlambdaReaperGraphSpec.v), as the queries go through them: every set
   of nodes that a query computes is a list, deduplicated, and every
   existential is over a list of facts or of tuples. The relations
   [field_of_constructor_is_used_top] and [field_of_constructor_is_used_as]
   of the post-processing rules of points_to_analysis.ml, which the
   tables have not, are computed from the facts and the tables by the
   rules that derive them ([field_used_top], [field_used_as]). The sets
   of closures of the unit ([sets_of_closures_by_function_slot]) are a
   list of sets, each the node and the code id of the closure at each of
   its function slots. The typing environment is the core's
   ([FlambdaTypingEnv.Typing_env.t]); the rewriter reads it through
   [Typing_env.find], [Typing_env.get_canonical_simple_exn],
   [expand_head] and [Provers.meet_single_closures_entry] of the core.

   The faults. Four rules of the compiler are not sound, and a flag of
   [fixes] gives each the fix proposed ([all_fixes]; [compiler_fixes] is
   the compiler, all off):
   - [fix_boxed]: a boxed number whose contents no use reads is made
     poison by the rebuild ([rebuild.ml], [reaper_unused_boxed_number]),
     and [Rewriter.rewrite] keeps its type ([Rule.identity] for a value
     that is not a closure). The fix forgets the type where
     [follow_field] of the field of the contents forgets it, as
     [rewrite_boxed_number_kind] does of the subkind.
   - [fix_alias]: under a metadata that forgets the type, an alias is
     still rewritten into one to the copy of its name for that metadata
     ([rewrite_arbitrary_type], [get_canonical_with]), and one copy
     stands for every position of the name under that metadata, though
     the rebuild may have made one of them poison and kept another. The
     fix rewrites such an alias to the unknown type.
   - [fix_refine]: where [refine_sets_of_closures] recovers the sources
     of a closure from its usages, it keeps only the sets whose closure
     is among them; [join_metadata] makes a closure that escapes
     ([usages_of] gives [None]) contribute no usage, so the recovered
     sources may miss its set. The fix keeps, there, the sets whose
     closure has any usage.
   - [fix_null]: the closure case of [Rewriter.rewrite] makes its type
     with [MTC.at_least_this_closure], which is not null; a closure type
     that may be null loses that. The fix keeps the nullability.

   The deliberate differences from the OCaml:
   - Fatal errors. A fatal error of the compiler, or an assertion that
     fails, is a case here like any other, and the model gives the
     unknown type there: [follow_field] of [No_source] or of
     [At_least_one_source_no_usages] gives
     [At_least_one_source_no_usages]; a [None] of the core's functions
     of types (a name not bound, a canonical element that is an alias)
     gives the unknown type; [X.set_of_closures] (a closure type that the
     rule leaves to the traversal) gives the unknown type.
   - The names of the copies. [get_canonical_with] names the copy of a
     name for a metadata while the type is rewritten. Here the aliases
     that the rewriting of a type reaches are listed first ([reached],
     which goes through the type as [rewrite] does), named in that
     order, and the type is then rewritten with the names
     ([rewrite], given [canon]). Fresh variables are numbered from
     [first_fresh], above every variable of the environment.
   - Fuel. The recursion of [rewrite] has fuel, the depth of the types
     of the environment and of the type, which is enough: a type is
     expanded at its head only, and only there is the type of another
     name read ([rewrite_fuel]). The rounds of [rewrite_in_depth] have
     fuel too; a copy left unprocessed has no equation, as the compiler
     gives one only to those it processes.
   - [Many_sources_usages] holds a list of nodes; two are the same
     metadata when they are the same set ([metadata_eqb]), as the
     compiler compares the keys of two maps.
   - [Function_slot.Map.choose] in [follow_field_for_set_of_closures]
     takes the first closure of the list.
   - The unboxed fields. Where [rewrite_result_types] or
     [patterns_for_unboxed_fields] stop with a fatal error, the leaves
     concerned have no variable, and their type is the unknown type of
     their kind: an unboxed variable with no allocation or no known use,
     a field that the unboxing has and that is not read, a field
     unboxed in turn that has not one source or is read into a name with
     any usage, and every field where the fields read are of distinct
     subkinds. [Tag.Map.get_singleton] of the tags of a block is a list
     of one binding; a [None] of [expand_head] and a head that is not of
     kind value (a fatal error) give the unknown type of the accessor.
     The uses of [add_usages_through_function_slots] through calls of
     known arity read [code_id_my_closure] off the term
     ([bound_code_my_closures]), as FlambdaReaperUnboxing.v does.

   What is left out:
   - In [rewrite], the sets of closures whose representation changes
     (unboxing). It takes the closures that change from [rc_changed];
     where one of the sets kept by [refine_sets_of_closures] changes,
     the compiler forgets the type if there are several or one may be
     external, and otherwise rewrites the type of the closure into one
     of its new representation ([Expr.exactly_this_closure] at the new
     slots, the value slots by [patterns_for_unboxed_fields] with the
     function slots); [rewrite] forgets the type in both cases. The
     compiler's rewriting is [rewrite_arbitrary_u] (below), which is not
     extracted.
   - [rewrite_kind_with_subkind]: the model has no subkinds.
   - In [rewrite_result_types], the environment of the result types is
     an input, where the compiler adds the extension to the environment
     of the unit by meet, and the extension it gives back is the
     equations and the existential copies, where the compiler makes it
     with [make_suitable_for_environment].
   - [Rec_info]: function types have no [rec_info] in the grammar, so the
     patterns [Pattern.rec_info] are not there; coercions are not there.
   - Array types and [array_slot]: the grammar has no [Array].
   - One compilation unit: every symbol, slot and code id is of the unit,
     so [has_local_fields] is that the closure type has a slot, and no
     name is left as it is for being of another unit. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaReaperGraphSpec.
From OxCamlFlambda2 Require FlambdaReaperUnboxing.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaBindingTime
  FlambdaTypingEnv FlambdaExpandHead FlambdaTypeAccessSpec FlambdaProvers.
Import ListNotations.

(* ======================================================================== *)
(* Fixes                                                                    *)
(* ======================================================================== *)

(* Which of the fixes of the header are on. *)
Record fixes : Type := Fixes {
  fix_boxed : bool;
  fix_alias : bool;
  fix_refine : bool;
  fix_null : bool
}.

(* The compiler. *)
Definition compiler_fixes : fixes := Fixes false false false false.

(* Every fix. *)
Definition all_fixes : fixes := Fixes true true true true.

(* ======================================================================== *)
(* The context                                                              *)
(* ======================================================================== *)

(* A set of closures: the node and the code id of the closure at each of
   its function slots. *)
Definition set_of_closures : Type :=
  list (function_slot * (node * Or_unknown.t code_id)).

(* [Types_rewriter.rewrite_context]: the facts of the graph, the tables
   of the solution, the local fields, the sets of closures of the unit,
   and the closures whose representation changes. *)
Record rewrite_context : Type := Rewrite_context {
  rc_facts : facts;
  rc_tables : tables;
  rc_is_local : field -> bool;
  rc_sets_of_closures : list set_of_closures;
  rc_changed : list node
}.

(* The context of the rebuild without unboxing: no closure changes. *)
Definition plain_context (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) : rewrite_context :=
  Rewrite_context F T is_local sets [].

(* ======================================================================== *)
(* Sets of nodes                                                            *)
(* ======================================================================== *)

(* The nodes of a list, each once. *)
Definition dedup_nodes (l : list node) : list node := Keyed.dedup node_eqb l.

(* Two lists of nodes are the same set. *)
Definition same_nodes (l l' : list node) : bool :=
  forallb (fun x => mem_node x l') l && forallb (fun x => mem_node x l) l'.

(* [Map.inter] of sets of nodes. *)
Definition inter_nodes (l l' : list node) : list node :=
  filter (fun x => mem_node x l') l.

(* [Map.subset_domain]. *)
Definition subset_nodes (l l' : list node) : bool :=
  forallb (fun x => mem_node x l') l.

Definition is_nil {A : Type} (l : list A) : bool :=
  match l with
  | [] => true
  | _ :: _ => false
  end.

(* ======================================================================== *)
(* Queries of the solution (points_to_analysis.ml)                          *)
(* ======================================================================== *)

Section Queries.

Variable ctx : rewrite_context.

Local Abbreviation F := (rc_facts ctx).
Local Abbreviation T := (rc_tables ctx).

Definition has_use (x : node) : bool := mem_node x (t_has_usage T).
Definition any_usage (x : node) : bool := mem_node x (t_any_usage T).
Definition any_source (x : node) : bool := mem_node x (t_any_source T).
Definition has_source (x : node) : bool := mem_node x (t_has_source T).

(* [field_used]: [field_of_constructor_is_used]. *)
Definition field_used (x : node) (f : field) : bool :=
  mem_node_field x f (t_field_used T).

(* The uses of a node, and its sources, as the tables have them; and the
   nodes of which it is a source. *)
Definition usages_of (x : node) : list node :=
  map snd (filter (fun p => node_eqb (fst p) x) (t_usages T)).

Definition sources_of (x : node) : list node :=
  map snd (filter (fun p => node_eqb (fst p) x) (t_sources T)).

Definition sourced_by (x : node) : list node :=
  map fst (filter (fun p => node_eqb (snd p) x) (t_sources T)).

(* The nodes that [base] holds in its field [f]. *)
Definition constructors_of (base : node) (f : field) : list node :=
  map (fun t => let '(_, _, from) := t in from)
    (filter (fun t => let '(b, f', _) := t in node_eqb b base && field_eqb f' f)
       (f_constructor F)).

(* The nodes that the field [f] of [base] is read into. *)
Definition accessors_of (f : field) (base : node) : list node :=
  map (fun t => let '(to_, _, _) := t in to_)
    (filter (fun t => let '(_, f', b) := t in field_eqb f' f && node_eqb b base)
       (f_accessor F)).

(* [get_direct_usages]: the uses, that have a use, of the nodes given. *)
Definition get_direct_usages (xs : list node) : list node :=
  dedup_nodes (flat_map (fun x => filter has_use (usages_of x)) xs).

(* [get_usages]. *)
Definition get_usages (x : node) : or_unknown_or_bottom (list node) :=
  if negb (has_use x) then Bottom
  else if any_usage x then Unknown
  else Ok (get_direct_usages [x]).

(* The sources of a node that have a source. *)
Definition allocation_points (x : node) : list node :=
  dedup_nodes (filter has_source (sources_of x)).

(* [get_single_source], through [multiple_allocation_points] and
   [allocation_point_dominator]: [Ok] of the one source, if there is one
   and the node has not any source. *)
Definition get_single_source (x : node) : or_unknown_or_bottom node :=
  if negb (has_source x) then Bottom
  else if any_source x then Unknown
  else
    match allocation_points x with
    | [y] => Ok y
    | _ => Unknown
    end.

(* [Points_to_analysis.single_field_source]. *)
Inductive single_field_source : Type :=
  | Field_no_source
  | Field_one_source (source : node)
  | Field_many_sources.

(* [get_single_field_source]. *)
Definition get_single_field_source (block : node) (f : field)
    : single_field_source :=
  let froms := constructors_of block f in
  if existsb any_source froms then Field_many_sources
  else
    match
      dedup_nodes (flat_map (fun from => filter has_source (sources_of from))
                     froms)
    with
    | [] => Field_no_source
    | [s] => Field_one_source s
    | _ => Field_many_sources
    end.

(* [get_one_field_usage]: what the field [f] of the uses given is read
   into. *)
Definition get_one_field_usage (f : field) (usages : list node)
    : or_unknown_or_bottom (list node) :=
  let readers := flat_map (accessors_of f) usages in
  if existsb any_usage readers then Unknown
  else
    match dedup_nodes (filter has_use readers) with
    | [] => Bottom
    | ys => Ok ys
    end.

(* [field_of_constructor_is_used_top], by the rules of
   [post_processing_rules] that derive it. *)
Definition field_used_top (base : node) (f : field) : bool :=
  negb (is_nil (constructors_of base f))
  && ((any_usage base && negb (rc_is_local ctx f))
      || existsb (fun u => existsb any_usage (accessors_of f u))
           (usages_of base)
      || (any_usage base
          && existsb (fun p => field_eqb (fst p) f && any_usage (snd p))
               (t_reading_field T))
      || (rc_is_local ctx f && any_usage base
          && existsb (fun u => existsb any_usage (accessors_of f u))
               (sourced_by base))).

(* [field_of_constructor_is_used_as], likewise: the nodes it holds of. *)
Definition field_used_as (base : node) (f : field) : list node :=
  if is_nil (constructors_of base f) then []
  else
    dedup_nodes
      (flat_map (fun u => filter has_use (accessors_of f u)) (usages_of base)
       ++ (if any_usage base
           then
             map snd
               (filter (fun p => field_eqb (fst p) f && has_use (snd p))
                  (t_reading_field T))
           else [])
       ++ (if rc_is_local ctx f && any_usage base
           then
             flat_map (fun u => filter has_use (accessors_of f u))
               (sourced_by base)
           else [])).

(* [get_one_field_usage_of_constructors]. *)
Definition get_one_field_usage_of_constructors (cs : list node) (f : field)
    : or_unknown_or_bottom (list node) :=
  if existsb (fun c => field_used_top c f) cs then Unknown
  else
    match dedup_nodes (flat_map (fun c => field_used_as c f) cs) with
    | [] => Bottom
    | ys => Ok ys
    end.

(* [Points_to_analysis.sources]. *)
Inductive allocation_sources : Type :=
  | Sources_any
  | Sources_known (sources : list node).

(* [get_direct_sources]. *)
Definition get_direct_sources (xs : list node) : allocation_sources :=
  if existsb any_source xs then Sources_any
  else
    Sources_known
      (dedup_nodes (flat_map (fun x => filter has_source (sources_of x)) xs)).

(* One round of the rules of
   [compute_usages_by_function_slots_not_following_known_arity_calls]:
   from a use [x] at a function slot, the uses of what a function slot
   [g] of [x] is read into, at [g]. *)
Definition function_slot_usages_step (out : list (function_slot * node))
    : list (function_slot * node) :=
  flat_map
    (fun p =>
       flat_map
         (fun t =>
            let '(y, f, b) := t in
            match f with
            | Field_function_slot g =>
              if node_eqb b (snd p)
              then map (fun z => (g, z)) (filter has_use (usages_of y))
              else []
            | _ => []
            end)
         (f_accessor F))
    out.

Definition mem_slot_node (p : function_slot * node)
    (l : list (function_slot * node)) : bool :=
  existsb (fun q => Pos.eqb (fst p) (fst q) && node_eqb (snd p) (snd q)) l.

(* The rounds, to the fixpoint. Each round that is not the last adds a
   pair of a function slot read by an accessor and of a use: the fuel of
   [compute_usages_by_function_slots] is their number. *)
Fixpoint function_slot_usages_fix (fuel : nat)
    (out : list (function_slot * node)) : list (function_slot * node) :=
  match fuel with
  | O => out
  | S fuel =>
    let next :=
      filter (fun p => negb (mem_slot_node p out))
        (function_slot_usages_step out)
    in
    if is_nil next then out
    else function_slot_usages_fix fuel (next ++ out)
  end.

(* [compute_usages_by_function_slots_not_following_known_arity_calls]. *)
Definition compute_usages_by_function_slots (usages : list node)
    (current : function_slot)
    : Or_unknown.t (list (function_slot * list node)) :=
  let out :=
    function_slot_usages_fix
      (S (length (f_accessor F) * length (t_usages T)))
      (map (fun x => (current, x)) usages)
  in
  if existsb (fun p => any_usage (snd p)) out then Or_unknown.Unknown
  else
    Or_unknown.Known
      (map
         (fun g =>
            (g, dedup_nodes
                  (map snd (filter (fun p => Pos.eqb (fst p) g) out))))
         (dedup (map fst out))).

End Queries.

(* ======================================================================== *)
(* Metadata                                                                 *)
(* ======================================================================== *)

(* [Types_rewriter.rewrite_metadata]: what is known of the sources and of
   the usages of the value that a type is of. *)
Inductive rewrite_metadata : Type :=
  | No_source
  | Single_source (source : node)
  | Many_sources_any_usage
  | Many_sources_usages (usages : list node)
  | At_least_one_source_no_usages.

(* [compare_metadata], as a test of equality. *)
Definition metadata_eqb (m m' : rewrite_metadata) : bool :=
  match m, m' with
  | No_source, No_source => true
  | Single_source s, Single_source s' => node_eqb s s'
  | Many_sources_any_usage, Many_sources_any_usage => true
  | Many_sources_usages u, Many_sources_usages u' => same_nodes u u'
  | At_least_one_source_no_usages, At_least_one_source_no_usages => true
  | _, _ => false
  end.

(* The metadata under which [Rewriter.rewrite] forgets the type:
   [At_least_one_source_no_usages], [Many_sources_usages] of no usage, and
   [No_source], which it makes unknown too. *)
Definition forgets (m : rewrite_metadata) : bool :=
  match m with
  | No_source | At_least_one_source_no_usages => true
  | Many_sources_usages usages => is_nil usages
  | Single_source _ | Many_sources_any_usage => false
  end.

Section Metadata.

Variable ctx : rewrite_context.

(* [for_usages] in [follow_field]. *)
Definition for_usages (usages : list node) (f : field) : rewrite_metadata :=
  match get_one_field_usage ctx f usages with
  | Unknown => Many_sources_any_usage
  | Bottom => At_least_one_source_no_usages
  | Ok vars => Many_sources_usages (get_direct_usages ctx vars)
  end.

(* [follow_field]: the metadata of what the field [f] holds. *)
Definition follow_field (m : rewrite_metadata) (f : field)
    : rewrite_metadata :=
  match m with
  | No_source | At_least_one_source_no_usages =>
    At_least_one_source_no_usages
  | Many_sources_any_usage => Many_sources_any_usage
  | Many_sources_usages usages => for_usages usages f
  | Single_source source =>
    if negb (field_used ctx source f) then At_least_one_source_no_usages
    else
      match get_single_field_source ctx source f with
      | Field_no_source => No_source
      | Field_one_source s => Single_source s
      | Field_many_sources =>
        match get_usages ctx source with
        | Unknown => Many_sources_any_usage
        | Bottom => At_least_one_source_no_usages
        | Ok usages => for_usages usages f
        end
      end
  end.

(* [usages_of] in [join_metadata]: [None] is unknown. *)
Definition usages_of_metadata (m : rewrite_metadata) : option (list node) :=
  match m with
  | Single_source source =>
    match get_usages ctx source with
    | Unknown => None
    | Ok usages => Some usages
    | Bottom => Some []
    end
  | Many_sources_any_usage => None
  | Many_sources_usages usages => Some usages
  | No_source | At_least_one_source_no_usages => Some []
  end.

Definition join_usages (x y : rewrite_metadata) : rewrite_metadata :=
  match usages_of_metadata x, usages_of_metadata y with
  | None, None => Many_sources_any_usage
  | Some u, None | None, Some u => Many_sources_usages u
  | Some u1, Some u2 => Many_sources_usages (inter_nodes u1 u2)
  end.

(* [join_metadata]. *)
Definition join_metadata (x y : rewrite_metadata) : rewrite_metadata :=
  match x, y with
  | No_source, _ => y
  | _, No_source => x
  | At_least_one_source_no_usages, _ | _, At_least_one_source_no_usages =>
    At_least_one_source_no_usages
  | Single_source s1, Single_source s2 =>
    if node_eqb s1 s2 then x else join_usages x y
  | _, _ => join_usages x y
  end.

(* [metadata_for_name]. *)
Definition metadata_for_name (n : node) : rewrite_metadata :=
  match get_single_source ctx n with
  | Bottom => No_source
  | Ok source =>
    if has_use ctx n then Single_source source
    else At_least_one_source_no_usages
  | Unknown =>
    match get_usages ctx n with
    | Bottom => At_least_one_source_no_usages
    | Unknown => Many_sources_any_usage
    | Ok usages => Many_sources_usages usages
    end
  end.

End Metadata.

(* The node of a name. *)
Definition node_of_name (n : name) : node :=
  match n with
  | Name_var x => Node_variable x
  | Name_symbol s => Node_symbol s
  end.

(* ======================================================================== *)
(* Closures (the closure case of [Rewriter.rewrite])                        *)
(* ======================================================================== *)

(* [Rewriter.must_be_local]. *)
Inductive must_be_local : Type :=
  | Must_be_local
  | Can_be_external.

(* The metadata of the value slots and of the function slots of a set of
   closures, each function slot with whether its code id is to be erased.
   *)
Definition slots_metadata : Type :=
  list (value_slot * rewrite_metadata)
  * list (function_slot * (rewrite_metadata * bool)).

(* [Map.inter] of maps keyed by slots. *)
Definition inter_slots {A B C : Type} (f : A -> B -> C) (l : finite_map A)
    (l' : finite_map B) : finite_map C :=
  flat_map
    (fun p =>
       match lookup (fst p) l' with
       | Some b => [(fst p, f (snd p) b)]
       | None => []
       end)
    (compact l).

Section Closures.

Variable fx : fixes.
Variable ctx : rewrite_context.

(* The closure of a set at a function slot. *)
Definition closure_at (set : set_of_closures) (g : function_slot) : node :=
  match lookup g set with
  | Some (closure, _) => closure
  | None => Node_all_constants
  end.

Definition intersect_all (sets : list (list node)) : list node :=
  match sets with
  | [] => []
  | first :: rest => fold_left inter_nodes rest first
  end.

(* [refine_sets_of_closures]: the sets of closures of the unit that the
   closure at [current] may come from, and whether it may also come
   from outside the unit. *)
Definition refine_sets_of_closures (t0 : rewrite_metadata)
    (current : function_slot)
    (code_ids_of_function_slots : list (function_slot * Or_unknown.t code_id))
    (mbl : must_be_local) : list set_of_closures * must_be_local :=
  let can_be_source_for_this_type (set : set_of_closures) :=
    forallb (fun p => is_bound (fst p) set) code_ids_of_function_slots
  in
  let candidates :=
    filter can_be_source_for_this_type
      (filter (fun set => is_bound current set) (rc_sets_of_closures ctx))
  in
  let for_known_sources (sources : list node) (recovered : bool) :=
    (filter
       (fun set =>
          mem_node (closure_at set current) sources
          || (recovered && fix_refine fx
              && any_usage ctx (closure_at set current)))
       candidates,
     Must_be_local)
  in
  let for_any_source_with_usages (usages_or_top : option (list node)) :=
    (filter
       (fun set =>
          let candidate_source := closure_at set current in
          any_usage ctx candidate_source
          || match usages_or_top with
             | None => false
             | Some usages =>
               subset_nodes usages
                 (get_direct_usages ctx [candidate_source])
             end)
       candidates,
     mbl)
  in
  match t0 with
  | No_source | At_least_one_source_no_usages => ([], mbl)
  | Many_sources_any_usage => for_any_source_with_usages None
  | Many_sources_usages s =>
    let usages_sources :=
      flat_map
        (fun usage =>
           match get_direct_sources ctx [usage] with
           | Sources_any => []
           | Sources_known sources => [sources]
           end)
        s
    in
    if is_nil usages_sources then for_any_source_with_usages (Some s)
    else for_known_sources (intersect_all usages_sources) true
  | Single_source source => for_known_sources [source] false
  end.

(* [follow_field_for_set_of_closures]; [None] is [raise Exit]. *)
Definition follow_field_for_set_of_closures (set : list (function_slot * node))
    (w : value_slot) : option rewrite_metadata :=
  let f := Field_value_slot w in
  if forallb (fun p => negb (field_used ctx (snd p) f)) set
  then Some At_least_one_source_no_usages
  else
    match set with
    | [] => Some At_least_one_source_no_usages
    | (_, c0) :: _ =>
      match get_single_field_source ctx c0 f with
      | Field_no_source => None
      | Field_one_source s => Some (Single_source s)
      | Field_many_sources =>
        match get_one_field_usage_of_constructors ctx (map snd set) f with
        | Unknown => Some Many_sources_any_usage
        | Bottom => Some At_least_one_source_no_usages
        | Ok vars => Some (Many_sources_usages (get_direct_usages ctx vars))
        end
      end
  end.

(* The metadata of the slots of a set of closures, from the set, or
   [None] if it is not a source ([from_local_sets_of_closures]). *)
Definition slots_of_local_set (value_slot_types : list value_slot)
    (function_slot_types : list function_slot) (set : set_of_closures)
    : option slots_metadata :=
  let nodes := map (fun p => (fst p, fst (snd p))) set in
  let fix value_slots (ws : list value_slot)
      : option (list (value_slot * rewrite_metadata)) :=
    match ws with
    | [] => Some []
    | w :: ws =>
      match follow_field_for_set_of_closures nodes w, value_slots ws with
      | None, _ | _, None => None
      | Some At_least_one_source_no_usages, Some rest => Some rest
      | Some m, Some rest => Some ((w, m) :: rest)
      end
    end
  in
  match value_slots value_slot_types with
  | None => None
  | Some vs =>
    Some
      (vs,
       map
         (fun g =>
            let source := closure_at set g in
            (g,
             (Single_source source,
              negb
                (field_used ctx source
                   (Field_call_witness Known_arity_code_pointer)
                 || field_used ctx source
                      (Field_call_witness Unknown_arity_code_pointer)))))
         function_slot_types)
  end.

(* [from_external_sources]. *)
Definition slots_of_external_sources (usages : rewrite_metadata)
    (current : function_slot) (value_slot_types : list value_slot)
    (function_slot_types : list function_slot) : option slots_metadata :=
  let any_usage :=
    Some
      (map (fun w => (w, Many_sources_any_usage)) value_slot_types,
       map (fun g => (g, (Many_sources_any_usage, false)))
         function_slot_types)
  in
  match usages with
  | Many_sources_usages usages =>
    match compute_usages_by_function_slots ctx usages current with
    | Or_unknown.Unknown => any_usage
    | Or_unknown.Known by_function_slot =>
      let all_usages := dedup_nodes (flat_map snd by_function_slot) in
      Some
        (map
           (fun w =>
              (w,
               follow_field ctx (Many_sources_usages all_usages)
                 (Field_value_slot w)))
           value_slot_types,
         flat_map
           (fun g =>
              match lookup g by_function_slot with
              | Some usages => [(g, (Many_sources_usages usages, false))]
              | None => []
              end)
           function_slot_types)
    end
  | _ => any_usage
  end.

(* The join of the metadata of the slots of two sets. *)
Definition join_slots (a b : slots_metadata) : slots_metadata :=
  (inter_slots (join_metadata ctx) (fst a) (fst b),
   inter_slots
     (fun x y => (join_metadata ctx (fst x) (fst y), snd x || snd y))
     (snd a) (snd b)).

(* What the closure case rewrites a closure type to: its function slot,
   its allocation mode, the code ids that it keeps, the metadata of the
   closures of its set and of its value slots, and whether it may be
   null (for [fix_null]). *)
Record closure_rewrite : Type := Closure_rewrite {
  cr_slot : function_slot;
  cr_alloc_mode : Alloc_mode.t;
  cr_function_types : list (function_slot * Or_unknown.t code_id);
  cr_closure_types : list (function_slot * rewrite_metadata);
  cr_value_slots : list (value_slot * rewrite_metadata);
  cr_maybe_null : bool
}.

(* What [Rewriter.rewrite] answers: [Rule.identity], [Rule.rewrite] to
   the unknown type, or the rule of a closure. *)
Inductive rule : Type :=
  | Rule_identity
  | Rule_forget
  | Rule_closure (spec : closure_rewrite).

(* The code id of a function slot that the entry has
   ([Flambda2_types.Closures_entry.find_function_type]). *)
Definition entry_code_id (entry : closures_entry) (g : function_slot)
    : Or_unknown.t code_id :=
  match closures_entry_find_function_type entry false g with
  | Ok ft => Or_unknown.Known (function_type_code_id ft)
  | Unknown | Bottom => Or_unknown.Unknown
  end.

(* The closure case of [Rewriter.rewrite], once
   [meet_single_closures_entry] has answered. *)
Definition closure_rule (usages : rewrite_metadata) (slot : function_slot)
    (alloc_mode : Alloc_mode.t) (entry : closures_entry) (maybe_null : bool)
    : rule :=
  match entry with
  | Closures_entry _ closure_types value_slot_types =>
    let value_slots := domain value_slot_types in
    let function_slots := domain closure_types in
    let code_ids := map (fun g => (g, entry_code_id entry g)) function_slots in
    let has_local_fields :=
      negb (is_nil value_slots) || negb (is_nil function_slots)
    in
    let mbl :=
      if has_local_fields
         || existsb
              (fun p =>
                 match snd p with
                 | Or_unknown.Known _ => true
                 | Or_unknown.Unknown => false
                 end)
              code_ids
      then Must_be_local
      else Can_be_external
    in
    let '(local_sets, mbl) :=
      refine_sets_of_closures usages slot code_ids mbl
    in
    if existsb
         (fun set =>
            existsb (fun p => mem_node (fst (snd p)) (rc_changed ctx)) set)
         local_sets
    then Rule_forget
    else
      let from_local :=
        flat_map
          (fun set =>
             match slots_of_local_set value_slots function_slots set with
             | Some s => [s]
             | None => []
             end)
          local_sets
      in
      let from_external :=
        match mbl with
        | Must_be_local => None
        | Can_be_external =>
          slots_of_external_sources usages slot value_slots function_slots
        end
      in
      let from_everything :=
        match from_external with
        | None => from_local
        | Some s => s :: from_local
        end
      in
      match from_everything with
      | [] => Rule_forget
      | first :: rest =>
        let '(vs, fs) := fold_left join_slots rest first in
        Rule_closure
          (Closure_rewrite slot alloc_mode
             (map
                (fun p : function_slot * (rewrite_metadata * bool) =>
                   (fst p,
                    if snd (snd p) then Or_unknown.Unknown
                    else entry_code_id entry (fst p)))
                fs)
             (map
                (fun p : function_slot * (rewrite_metadata * bool) =>
                   (fst p, fst (snd p)))
                fs)
             vs maybe_null)
      end
  end.

(* Whether the head of a type, expanded, may be null. *)
Definition expanded_maybe_null (tenv : Typing_env.t) (ty : flambda_type)
    : bool :=
  match expand_head tenv ty with
  | Some expanded =>
    match Expanded_type.descr expanded with
    | Ok (Expanded_type.Head_value (Head_of_kind_value _ (Maybe_null _))) =>
      true
    | _ => false
    end
  | None => false
  end.

(* [Rewriter.rewrite]. *)
Definition rewrite_rule (tenv : Typing_env.t) (usages : rewrite_metadata)
    (ty : flambda_type) : rule :=
  if forgets usages then Rule_forget
  else
    match kind ty with
    | Rec_info | Region | Naked_number _ => Rule_identity
    | Value =>
      match Provers.meet_single_closures_entry tenv ty with
      | Some Meet_shortcut.Invalid => Rule_identity
      | Some Meet_shortcut.Need_meet | None => Rule_forget
      | Some (Meet_shortcut.Known_result (slot, alloc_mode, entry, _)) =>
        closure_rule usages slot alloc_mode entry
          (expanded_maybe_null tenv ty)
      end
    end.

End Closures.

(* ======================================================================== *)
(* The traversal ([Traversals.Make])                                        *)
(* ======================================================================== *)

(* [destructure_expanded_head] at the discriminant [Closure], for the
   accessor of a value slot ([get_env_var]) or of a function slot
   ([get_closure]). The kind of a value slot is not in the model; the
   unknown type there is at kind value, and is not reached, as the
   value slots destructured are those of the entry. *)
Definition destructure_closure
    (get : row_like_for_closures -> option (Or_unknown.t flambda_type))
    (expanded : option Expanded_type.t) : flambda_type :=
  match expanded with
  | None => unknown Value
  | Some expanded =>
    match Expanded_type.descr expanded with
    | Unknown => unknown Value
    | Bottom => bottom Value
    | Ok (Expanded_type.Head_value (Head_of_kind_value non_null _)) =>
      match non_null with
      | Unknown => unknown Value
      | Bottom => bottom Value
      | Ok (Closures by_function_slot _) =>
        match get by_function_slot with
        | Some (Or_unknown.Known ty) => ty
        | Some Or_unknown.Unknown | None => unknown Value
        end
      | Ok _ => bottom Value
      end
    | Ok _ => unknown Value
    end
  end.

(* A copy of a name for a metadata, at a kind: what [get_canonical_with]
   gives. *)
Definition canon_fn : Type := name -> flambda_kind -> rewrite_metadata -> name.

Section Rewrite.

Variable fx : fixes.
Variable ctx : rewrite_context.
Variable tenv : Typing_env.t.
Variable canon : canon_fn.

(* [block_slot]: the metadata of the field [i] of a block whose field
   type is [ty]. *)
Definition block_slot (m : rewrite_metadata) (i : nat) (ty : flambda_type)
    : rewrite_metadata :=
  follow_field ctx m (Field_block i (kind ty)).

(* [rewrite_int_indexed_product] with [block_slot]. The argument [rw]
   stands for [rewrite_arbitrary_type] in this and the definitions that
   follow, as the recursion is through it. *)
Definition rewrite_fields
    (rw : rewrite_metadata -> flambda_type -> flambda_type)
    (m : rewrite_metadata) (tys : list flambda_type) : list flambda_type :=
  map (fun p => rw (block_slot m (fst p) (snd p)) (snd p))
    (combine (seq 0 (length tys)) tys).

(* [rewrite_row_like_for_blocks]: the fields of each case, the
   environment extensions dropped. *)
Definition rewrite_block_case
    (rw : rewrite_metadata -> flambda_type -> flambda_type)
    (m : rewrite_metadata) (case : row_like_block_case) : row_like_block_case :=
  Row_like_case (rewrite_fields rw m (case_maps_to case)) (case_index case)
    empty_env_extension.

Definition rewrite_row_like_for_blocks
    (rw : rewrite_metadata -> flambda_type -> flambda_type)
    (m : rewrite_metadata) (blocks : row_like_for_blocks)
    : row_like_for_blocks :=
  match blocks with
  | Row_like_for_blocks known_tags other_tags alloc_mode =>
    Row_like_for_blocks
      (Keyed.map_values
         (fun case =>
            match case with
            | Or_unknown.Unknown => Or_unknown.Unknown
            | Or_unknown.Known case =>
              Or_unknown.Known (rewrite_block_case rw m case)
            end)
         known_tags)
      match other_tags with
      | Or_bottom.Bottom => Or_bottom.Bottom
      | Or_bottom.Ok case => Or_bottom.Ok (rewrite_block_case rw m case)
      end
      alloc_mode
  end.

(* [rewrite_head_of_kind_value_non_null]: the variables [is_int] and
   [get_tag] and the extensions of a variant are dropped. A closure type
   is the rule's to rewrite: if it gets here, the compiler stops in
   [X.set_of_closures], unless the structure has no case. *)
Definition rewrite_head_of_kind_value_non_null
    (rw : rewrite_metadata -> flambda_type -> flambda_type)
    (m : rewrite_metadata) (head : head_of_kind_value_non_null)
    : or_unknown_or_bottom head_of_kind_value_non_null :=
  match head with
  | Variant _ immediates _ blocks _ is_unique =>
    Ok
      (Variant None
         match immediates with
         | Or_unknown.Unknown => Or_unknown.Unknown
         | Or_unknown.Known ty => Or_unknown.Known (rw m ty)
         end
         None
         match blocks with
         | Or_unknown.Unknown => Or_unknown.Unknown
         | Or_unknown.Known blocks =>
           Or_unknown.Known (rewrite_row_like_for_blocks rw m blocks)
         end
         No_extensions is_unique)
  | Mutable_block _ | String _ => Ok head
  | Boxed_number k contents alloc_mode =>
    if fix_boxed fx && forgets (follow_field ctx m (Field_boxed_number k))
    then Unknown
    else Ok (Boxed_number k (rw m contents) alloc_mode)
  | Closures (Row_like_for_closures []) _ => Ok head
  | Closures _ _ => Unknown
  end.

(* [rewrite_head_of_kind_value]: the variable [is_null] is dropped. *)
Definition rewrite_head_of_kind_value
    (rw : rewrite_metadata -> flambda_type -> flambda_type)
    (m : rewrite_metadata) (head : head_of_kind_value) : head_of_kind_value :=
  match head with
  | Head_of_kind_value non_null is_null =>
    Head_of_kind_value
      match non_null with
      | Unknown => Unknown
      | Bottom => Bottom
      | Ok head => rewrite_head_of_kind_value_non_null rw m head
      end
      match is_null with
      | Not_null => Not_null
      | Maybe_null _ => Maybe_null None
      end
  end.

(* [rewrite_expanded_head], then [Expanded_type.to_type]: the relations
   of a naked immediate are dropped. *)
Definition rewrite_expanded_head
    (rw : rewrite_metadata -> flambda_type -> flambda_type)
    (m : rewrite_metadata) (expanded : Expanded_type.t) : flambda_type :=
  match Expanded_type.descr expanded with
  | Unknown => unknown (Expanded_type.kind expanded)
  | Bottom => bottom (Expanded_type.kind expanded)
  | Ok (Expanded_type.Head_value head) =>
    Type_value (Ok (No_alias (rewrite_head_of_kind_value rw m head)))
  | Ok (Expanded_type.Head_naked_immediate head) =>
    Type_naked_immediate
      (Ok (No_alias (Head_of_kind_naked_immediate (naked_immediates head) [])))
  | Ok (Expanded_type.Head_naked_number k numbers) =>
    Type_naked_number k (Ok (No_alias numbers))
  | Ok (Expanded_type.Head_rec_info head) => Type_rec_info (Ok (No_alias head))
  | Ok (Expanded_type.Head_region head) => Type_region (Ok (No_alias head))
  end.

(* A type that is not an alias, rewritten as [rewrite] does under
   [Rule.identity]: a type that is a constant becomes an alias to it. *)
Definition rewrite_identity
    (rw : rewrite_metadata -> flambda_type -> flambda_type)
    (m : rewrite_metadata) (ty : flambda_type) : flambda_type :=
  match expand_head tenv ty with
  | None => unknown (kind ty)
  | Some expanded =>
    let ty' := rewrite_expanded_head rw m expanded in
    match must_be_singleton ty' with
    | Some (Some c) => alias_type_of (kind ty') (Const c)
    | _ => ty'
    end
  end.

(* The rule of a closure, applied: [match_pattern] and [rewrite_expr] for
   [Expr.at_least_this_closure]. *)
Definition rewrite_closure
    (rw : rewrite_metadata -> flambda_type -> flambda_type)
    (spec : closure_rewrite) (ty : flambda_type) : flambda_type :=
  let expanded := expand_head tenv ty in
  let ty' :=
    at_least_this_closure (cr_slot spec)
      (map
         (fun p =>
            (fst p,
             match snd p with
             | Or_unknown.Known code => Or_unknown.Known (Function_type code)
             | Or_unknown.Unknown => Or_unknown.Unknown
             end))
         (cr_function_types spec))
      (map
         (fun p =>
            (fst p,
             rw (snd p)
               (destructure_closure
                  (fun c => row_like_for_closures_get_closure c (fst p))
                  expanded)))
         (cr_closure_types spec))
      (map
         (fun p =>
            (fst p,
             rw (snd p)
               (destructure_closure
                  (fun c => row_like_for_closures_get_env_var c (fst p))
                  expanded)))
         (cr_value_slots spec))
      (cr_alloc_mode spec)
  in
  if fix_null fx && cr_maybe_null spec
  then
    match ty' with
    | Type_value (Ok (No_alias (Head_of_kind_value non_null _))) =>
      Type_value (Ok (No_alias (Head_of_kind_value non_null (Maybe_null None))))
    | _ => ty'
    end
  else ty'.

(* [rewrite_arbitrary_type] and [rewrite]. *)
Fixpoint rewrite_arbitrary (fuel : nat) (m : rewrite_metadata)
    (ty : flambda_type) {struct fuel} : flambda_type :=
  match fuel with
  | O => unknown (kind ty)
  | S fuel =>
    match get_alias_opt ty with
    | Some alias =>
      match
        Typing_env.get_canonical_simple_exn tenv (Some Name_mode.in_types)
          None alias
      with
      | Some (Some (Const c)) => rewrite fuel m (type_for_const c)
      | Some (Some (Var x)) =>
        if fix_alias fx && forgets m then unknown (kind ty)
        else
          alias_type_of (kind ty)
            (simple_of_name (canon (Name_var x) (kind ty) m))
      | Some (Some (Symbol s)) =>
        if fix_alias fx && forgets m then unknown (kind ty)
        else
          alias_type_of (kind ty)
            (simple_of_name (canon (Name_symbol s) (kind ty) m))
      | Some None | None => unknown (kind ty)
      end
    | None => rewrite fuel m ty
    end
  end

with rewrite (fuel : nat) (m : rewrite_metadata) (ty : flambda_type)
    {struct fuel} : flambda_type :=
  match fuel with
  | O => unknown (kind ty)
  | S fuel =>
    match rewrite_rule fx ctx tenv m ty with
    | Rule_forget => unknown (kind ty)
    | Rule_identity => rewrite_identity (rewrite_arbitrary fuel) m ty
    | Rule_closure spec => rewrite_closure (rewrite_arbitrary fuel) spec ty
    end
  end.

End Rewrite.

(* ======================================================================== *)
(* The aliases that a rewriting reaches                                     *)
(* ======================================================================== *)

(* A name at a kind under a metadata: what [get_canonical_with] is asked
   for. *)
Definition reach : Type := name * flambda_kind * rewrite_metadata.

Section Reached.

Variable fx : fixes.
Variable ctx : rewrite_context.
Variable tenv : Typing_env.t.

(* What [rewrite_fields] and the functions after it reach, given what
   [rw] reaches. *)
Definition reached_fields (rw : rewrite_metadata -> flambda_type -> list reach)
    (m : rewrite_metadata) (tys : list flambda_type) : list reach :=
  flat_map (fun p => rw (block_slot ctx m (fst p) (snd p)) (snd p))
    (combine (seq 0 (length tys)) tys).

Definition reached_row_like_for_blocks
    (rw : rewrite_metadata -> flambda_type -> list reach)
    (m : rewrite_metadata) (blocks : row_like_for_blocks) : list reach :=
  match blocks with
  | Row_like_for_blocks known_tags other_tags _ =>
    flat_map
      (fun p =>
         match snd p with
         | Or_unknown.Unknown => []
         | Or_unknown.Known case => reached_fields rw m (case_maps_to case)
         end)
      known_tags
    ++ match other_tags with
       | Or_bottom.Bottom => []
       | Or_bottom.Ok case => reached_fields rw m (case_maps_to case)
       end
  end.

Definition reached_head_of_kind_value_non_null
    (rw : rewrite_metadata -> flambda_type -> list reach)
    (m : rewrite_metadata) (head : head_of_kind_value_non_null) : list reach :=
  match head with
  | Variant _ immediates _ blocks _ _ =>
    match blocks with
    | Or_unknown.Unknown => []
    | Or_unknown.Known blocks => reached_row_like_for_blocks rw m blocks
    end
    ++ match immediates with
       | Or_unknown.Unknown => []
       | Or_unknown.Known ty => rw m ty
       end
  | Mutable_block _ | String _ | Closures _ _ => []
  | Boxed_number k contents _ =>
    if fix_boxed fx && forgets (follow_field ctx m (Field_boxed_number k))
    then []
    else rw m contents
  end.

Definition reached_identity
    (rw : rewrite_metadata -> flambda_type -> list reach)
    (m : rewrite_metadata) (ty : flambda_type) : list reach :=
  match expand_head tenv ty with
  | Some expanded =>
    match Expanded_type.descr expanded with
    | Ok (Expanded_type.Head_value (Head_of_kind_value (Ok head) _)) =>
      reached_head_of_kind_value_non_null rw m head
    | _ => []
    end
  | None => []
  end.

Definition reached_closure (rw : rewrite_metadata -> flambda_type -> list reach)
    (spec : closure_rewrite) (ty : flambda_type) : list reach :=
  let expanded := expand_head tenv ty in
  flat_map
    (fun p =>
       rw (snd p)
         (destructure_closure
            (fun c => row_like_for_closures_get_closure c (fst p)) expanded))
    (cr_closure_types spec)
  ++ flat_map
       (fun p =>
          rw (snd p)
            (destructure_closure
               (fun c => row_like_for_closures_get_env_var c (fst p))
               expanded))
       (cr_value_slots spec).

(* The aliases that [rewrite_arbitrary] and [rewrite] give to [canon],
   in the order of the compiler's traversal. *)
Fixpoint reached_arbitrary (fuel : nat) (m : rewrite_metadata)
    (ty : flambda_type) {struct fuel} : list reach :=
  match fuel with
  | O => []
  | S fuel =>
    match get_alias_opt ty with
    | Some alias =>
      match
        Typing_env.get_canonical_simple_exn tenv (Some Name_mode.in_types)
          None alias
      with
      | Some (Some (Const c)) => reached fuel m (type_for_const c)
      | Some (Some (Var x)) =>
        if fix_alias fx && forgets m then [] else [(Name_var x, kind ty, m)]
      | Some (Some (Symbol s)) =>
        if fix_alias fx && forgets m then []
        else [(Name_symbol s, kind ty, m)]
      | Some None | None => []
      end
    | None => reached fuel m ty
    end
  end

with reached (fuel : nat) (m : rewrite_metadata) (ty : flambda_type)
    {struct fuel} : list reach :=
  match fuel with
  | O => []
  | S fuel =>
    match rewrite_rule fx ctx tenv m ty with
    | Rule_forget => []
    | Rule_identity => reached_identity (reached_arbitrary fuel) m ty
    | Rule_closure spec => reached_closure (reached_arbitrary fuel) spec ty
    end
  end.

End Reached.

(* ======================================================================== *)
(* Fuel                                                                     *)
(* ======================================================================== *)

(* The greatest of a list of numbers. *)
Definition list_max (l : list nat) : nat := fold_right Nat.max 0 l.

(* The depth of a type: of nesting of types in types. *)
Fixpoint type_depth (t : flambda_type) : nat :=
  match t with
  | Type_value (Ok (No_alias (Head_of_kind_value (Ok head) _))) =>
    S (non_null_depth head)
  | Type_value _ | Type_naked_immediate _ | Type_naked_number _ _
  | Type_rec_info _ | Type_region _ => 1
  end

with non_null_depth (head : head_of_kind_value_non_null) : nat :=
  match head with
  | Variant _ immediates _ blocks _ _ =>
    S (Nat.max
         match immediates with
         | Or_unknown.Known ty => type_depth ty
         | Or_unknown.Unknown => 0
         end
         match blocks with
         | Or_unknown.Known blocks => blocks_depth blocks
         | Or_unknown.Unknown => 0
         end)
  | Boxed_number _ contents _ => S (type_depth contents)
  | Closures closures _ => S (closures_depth closures)
  | Mutable_block _ | String _ => 1
  end

with blocks_depth (blocks : row_like_for_blocks) : nat :=
  match blocks with
  | Row_like_for_blocks known_tags other_tags _ =>
    Nat.max
      (list_max
         (map
            (fun p =>
               match snd p with
               | Or_unknown.Known case =>
                 list_max (map type_depth (case_maps_to case))
               | Or_unknown.Unknown => 0
               end)
            known_tags))
      match other_tags with
      | Or_bottom.Ok case => list_max (map type_depth (case_maps_to case))
      | Or_bottom.Bottom => 0
      end
  end

with closures_depth (closures : row_like_for_closures) : nat :=
  match closures with
  | Row_like_for_closures known_closures =>
    list_max
      (map
         (fun p =>
            match case_maps_to (snd p) with
            | Closures_entry _ closure_types value_slot_types =>
              Nat.max
                (list_max (map (fun q => type_depth (snd q)) closure_types))
                (list_max (map (fun q => type_depth (snd q)) value_slot_types))
            end)
         known_closures)
  end.

(* The deepest type of the environment. *)
Definition env_depth (tenv : Typing_env.t) : nat :=
  list_max (map (fun p => type_depth (fst (snd p)))
              (Typing_env.names_to_types tenv)).

(* The fuel of the rewriting of a type: its depth, or that of the type
   it is expanded to at its head, and a step for each of the two
   functions at each level, and for the type of a constant. *)
Definition rewrite_fuel (tenv : Typing_env.t) (ty : flambda_type) : nat :=
  2 * (type_depth ty + env_depth tenv) + 4.

(* ======================================================================== *)
(* The copies of names ([get_canonical_with])                               *)
(* ======================================================================== *)

(* The accumulator of [Traversals.Make]: the copies of each name, by
   metadata, the copies yet to be given an equation (each with the name
   it copies, its metadata and its kind), and the next fresh variable. *)
Record names_acc : Type := Names_acc {
  aliases_of_names : Keyed.finite_map name (list (rewrite_metadata * name));
  names_to_process : list (name * rewrite_metadata * flambda_kind * name);
  next_fresh : positive
}.

Fixpoint lookup_metadata {A : Type} (m : rewrite_metadata)
    (l : list (rewrite_metadata * A)) : option A :=
  match l with
  | [] => None
  | (m', a) :: l => if metadata_eqb m m' then Some a else lookup_metadata m l
  end.

(* [get_canonical_with]: the first metadata of a name has the name
   itself, each other a fresh variable. *)
Definition get_canonical_with (acc : names_acc) (r : reach) : names_acc :=
  let '(canonical, k, m) := r in
  match Keyed.lookup name_eqb canonical (aliases_of_names acc) with
  | None =>
    Names_acc ((canonical, [(m, canonical)]) :: aliases_of_names acc)
      ((canonical, m, k, canonical) :: names_to_process acc) (next_fresh acc)
  | Some aliases =>
    match lookup_metadata m aliases with
    | Some _ => acc
    | None =>
      let x' := Name_var (next_fresh acc) in
      Names_acc ((canonical, (m, x') :: aliases) :: aliases_of_names acc)
        ((canonical, m, k, x') :: names_to_process acc)
        (Pos.succ (next_fresh acc))
    end
  end.

(* The copy of a name for a metadata that the accumulator has, or the
   name itself. *)
Definition canon_of (acc : names_acc) : canon_fn :=
  fun n _ m =>
    match Keyed.lookup name_eqb n (aliases_of_names acc) with
    | Some aliases =>
      match lookup_metadata m aliases with
      | Some n' => n'
      | None => n
      end
    | None => n
    end.

(* [Points_to_analysis.keep_or_delete]: what the change of calling
   convention does with a parameter or a result of code. *)
Inductive keep_or_delete : Type :=
  | Keep
  | Delete.

(* The result types of code, rewritten ([Result_types.t], as
   [rewrite_result_types] makes it): the new variables of the parameters
   and of the results kept, the variables of the copies, which are
   existential, and the equations. *)
Record rewritten_result_types : Type := Rewritten_result_types {
  rr_params : list variable;
  rr_results : list variable;
  rr_existentials : list variable;
  rr_equations : list (name * flambda_type)
}.

(* The elements of a list, each with a number, from [first]. *)
Fixpoint number_from {A : Type} (first : positive) (l : list A)
    : list (positive * A) :=
  match l with
  | [] => []
  | a :: l => (first, a) :: number_from (Pos.succ first) l
  end.

(* ======================================================================== *)
(* The typing environment ([rewrite_in_depth], [rewrite])                   *)
(* ======================================================================== *)

Section Env.

Variable fx : fixes.
Variable ctx : rewrite_context.
Variable tenv : Typing_env.t.

(* [rewrite_concrete_type_of]: the type of a name, rewritten. *)
Definition concrete_type_of (n : name) (k : flambda_kind) : flambda_type :=
  match Typing_env.find tenv n (Some k) with
  | Some ty => ty
  | None => unknown k
  end.

Definition rewrite_concrete_type_of (canon : canon_fn) (n : name)
    (k : flambda_kind) (m : rewrite_metadata) : flambda_type :=
  let ty := concrete_type_of n k in
  rewrite fx ctx tenv canon (rewrite_fuel tenv ty) m ty.

Definition reached_concrete_type_of (n : name) (k : flambda_kind)
    (m : rewrite_metadata) : list reach :=
  let ty := concrete_type_of n k in
  reached fx ctx tenv (rewrite_fuel tenv ty) m ty.

(* [rewrite_in_depth]: each round names the copies that the types of the
   names to process reach, then rewrites these types with the names. *)
Fixpoint rewrite_in_depth (fuel : nat) (acc : names_acc)
    (new_types : list (name * flambda_type))
    : list (name * flambda_type) * names_acc :=
  match fuel with
  | O => (new_types, acc)
  | S fuel =>
    match names_to_process acc with
    | [] => (new_types, acc)
    | to_process =>
      let acc0 := Names_acc (aliases_of_names acc) [] (next_fresh acc) in
      let acc1 :=
        fold_left
          (fun acc t =>
             let '(before, m, k, _) := t in
             fold_left get_canonical_with (reached_concrete_type_of before k m)
               acc)
          to_process acc0
      in
      let types :=
        map
          (fun t =>
             let '(before, m, k, after) := t in
             (after, rewrite_concrete_type_of (canon_of acc1) before k m))
          to_process
      in
      rewrite_in_depth fuel acc1 (new_types ++ types)
    end
  end.

(* A variable above every variable of the environment. *)
Definition first_fresh_of_env : positive :=
  Pos.succ
    (fold_right Pos.max 1%positive
       (flat_map
          (fun p =>
             match fst p with
             | Name_var x => [x]
             | Name_symbol _ => []
             end
             ++ free_variables (fst (snd p)))
          (Typing_env.names_to_types tenv))).

(* The rounds of [rewrite_in_depth]: one for each name of the
   environment, and one more. *)
Definition rounds_of_env : nat :=
  S (length (Typing_env.names_to_types tenv)
     + length (Typing_env.defined_symbols tenv)).

(* [rewrite], the function of [Traversals.Make] that
   [Types_rewriter.rewrite_typing_env] calls: each symbol is processed
   with its metadata, under its own name. The equations of the
   environment of the .cmx file, one for each name processed (the
   compiler adds them to an environment where the symbols and the
   copies are defined). *)
Definition rewrite_typing_env : list (name * flambda_type) :=
  let symbols := Typing_env.defined_symbols tenv in
  let acc :=
    Names_acc
      (map
         (fun s =>
            (Name_symbol s,
             [(metadata_for_name ctx (Node_symbol s), Name_symbol s)]))
         symbols)
      (map
         (fun s =>
            (Name_symbol s, metadata_for_name ctx (Node_symbol s), Value,
             Name_symbol s))
         symbols)
      first_fresh_of_env
  in
  fst (rewrite_in_depth rounds_of_env acc []).

(* The type of a symbol, rewritten as [rewrite_typing_env] rewrites it,
   given the copies of the names it reaches. *)
Definition rewrite_symbol_type (canon : canon_fn) (s : symbol)
    : flambda_type :=
  rewrite_concrete_type_of canon (Name_symbol s) Value
    (metadata_for_name ctx (Node_symbol s)).

(* ======================================================================== *)
(* The result types of code ([rewrite_result_types])                       *)
(* ======================================================================== *)

(* The kind of a name, read off its type ([Variable.kind]). *)
Definition kind_of_name (n : name) : flambda_kind :=
  match Typing_env.find tenv n None with
  | Some ty => kind ty
  | None => Value
  end.

(* The type of the copy that [rewrite_result_types] makes of a parameter
   or a result that the code keeps: the type that the result types give
   the parameter of the result types [type_var], read in the environment
   of the result types, rewritten under the metadata of the variable of
   the code [code_var] ([variable_pattern], [Keep]). *)
Definition rewrite_result_type (canon : canon_fn) (code_var type_var : variable)
    : flambda_type :=
  rewrite_concrete_type_of canon (Name_var type_var)
    (kind_of_name (Name_var type_var))
    (metadata_for_name ctx (Node_variable code_var)).

(* The alias that the new variable of a parameter or a result kept is
   given, before it is rewritten: [alias_type_of kind typev]. *)
Definition result_alias (type_var : variable) : flambda_type :=
  alias_type_of (kind_of_name (Name_var type_var)) (Var type_var).

(* [rewrite_result_types], without unboxing. [tenv] is the environment of
   the result types: the environment of the unit at the end, with the
   parameters and the results of the result types defined and the
   equations of their extension added. Each parameter and each result is
   given with the variable of the code and the decision of the change of
   calling convention ([Keep] everywhere when it does not change); one
   that is deleted is dropped, and one that is kept gets a new variable,
   numbered from [first_fresh], whose type is the alias to it rewritten
   under the metadata of the variable of the code
   ([rewrite_env_extension_with_extra_variables]: [fold_destructuring] of
   [Pattern.var], then [rewrite_arbitrary_type]); the copies that these
   reach are processed as those of the environment ([rewrite_in_depth]),
   and are the existential variables of the extension. *)
Definition rewrite_result_types
    (params results : list (variable * keep_or_delete * variable))
    : rewritten_result_types :=
  let kept (l : list (variable * keep_or_delete * variable)) :=
    flat_map
      (fun t =>
         let '(code_var, decision, type_var) := t in
         match decision with
         | Keep => [(code_var, type_var)]
         | Delete => []
         end)
      l
  in
  let kept_params := kept params in
  let kept_results := kept results in
  let numbered := number_from first_fresh_of_env (kept_params ++ kept_results)
  in
  let next := Pos.of_nat (Pos.to_nat first_fresh_of_env + length numbered) in
  let acc :=
    fold_left
      (fun acc t =>
         let '(_, (code_var, type_var)) := t in
         let ty := result_alias type_var in
         fold_left get_canonical_with
           (reached_arbitrary fx ctx tenv (rewrite_fuel tenv ty)
              (metadata_for_name ctx (Node_variable code_var)) ty)
           acc)
      numbered (Names_acc [] [] next)
  in
  let bindings :=
    map
      (fun t =>
         let '(new_var, (code_var, type_var)) := t in
         let ty := result_alias type_var in
         (Name_var new_var,
          rewrite_arbitrary fx ctx tenv (canon_of acc) (rewrite_fuel tenv ty)
            (metadata_for_name ctx (Node_variable code_var)) ty))
      numbered
  in
  let '(copies, _) := rewrite_in_depth rounds_of_env acc [] in
  Rewritten_result_types (map fst (firstn (length kept_params) numbered))
    (map fst (skipn (length kept_params) numbered))
    (flat_map
       (fun p => match fst p with Name_var x => [x] | Name_symbol _ => [] end)
       copies)
    (bindings ++ copies).

End Env.

(* ======================================================================== *)
(* Unboxed parameters and results ([patterns_for_unboxed_fields])          *)
(* ======================================================================== *)

(* [Field.kind]. *)
Local Abbreviation field_kind := FlambdaReaperUnboxing.field_kind.

(* The paths of the leaves of an unboxing, each with the kind of its last
   field, in the order of [Unboxed_fields.fold_with_kind]. *)
Fixpoint unboxed_field_paths (kind : flambda_kind)
    (u : FlambdaReaperUnboxing.unboxed_field) {struct u}
    : list (list field * flambda_kind) :=
  match u with
  | FlambdaReaperUnboxing.Not_unboxed _ => [([], kind)]
  | FlambdaReaperUnboxing.Unboxed fields =>
    (fix paths (fields : list (field * FlambdaReaperUnboxing.unboxed_field))
         : list (list field * flambda_kind) :=
       match fields with
       | [] => []
       | (f, u) :: fields =>
         map (fun pk => (f :: fst pk, snd pk))
           (unboxed_field_paths (field_kind f) u)
         ++ paths fields
       end) fields
  end.

(* The classification of the fields of an unboxing
   ([classify_field_map]): the indices of the fields of a block met so
   far. *)
Inductive field_class : Type :=
  | Class_empty
  | Class_distinct_subkinds
  | Class_boxed_number
  | Class_block (indices : list nat)
  | Class_closure.

Definition classify_step (acc : field_class) (f : field) : field_class :=
  match acc with
  | Class_distinct_subkinds | Class_boxed_number => Class_distinct_subkinds
  | Class_empty | Class_block _ | Class_closure =>
    match f with
    | Field_call_witness _ | Field_return_of_call _
    | Field_code_id_of_call_witness =>
      Class_distinct_subkinds
    | Field_boxed_number _ =>
      match acc with
      | Class_empty => Class_boxed_number
      | _ => Class_distinct_subkinds
      end
    | Field_value_slot _ | Field_function_slot _ =>
      match acc with
      | Class_empty | Class_closure => Class_closure
      | _ => Class_distinct_subkinds
      end
    | Field_is_int | Field_get_tag =>
      match acc with
      | Class_empty => Class_block []
      | Class_block indices => Class_block indices
      | _ => Class_distinct_subkinds
      end
    | Field_block i _ =>
      match acc with
      | Class_empty => Class_block [i]
      | Class_block indices =>
        if existsb (Nat.eqb i) indices then Class_distinct_subkinds
        else Class_block (i :: indices)
      | _ => Class_distinct_subkinds
      end
    end
  end.

Definition classify_field_map (fields : list field) : field_class :=
  fold_left classify_step fields Class_empty.

(* The classes under which [patterns_for_unboxed_fields] binds no
   variable ([forget]). *)
Definition forgets_class (c : field_class) : bool :=
  match c with
  | Class_empty | Class_distinct_subkinds => true
  | Class_boxed_number | Class_block _ | Class_closure => false
  end.

(* A leaf of the pattern of an unboxing: its path, its kind, and the
   metadata of its variable, [None] where the compiler binds none. *)
Definition unboxed_leaf : Type :=
  list field * flambda_kind * option rewrite_metadata.

(* [forget]: the leaves of a field, with no variable. *)
Definition forgotten_leaves (f : field)
    (u : FlambdaReaperUnboxing.unboxed_field) : list unboxed_leaf :=
  map (fun pk => (f :: fst pk, snd pk, None))
    (unboxed_field_paths (field_kind f) u).

Section Unboxed.

Variable fx : fixes.
Variable ctx : rewrite_context.
Variable tenv : Typing_env.t.
(* The relation [code_id_my_closure] (FlambdaReaperUnboxing.v,
   [bound_code_my_closures]). *)
Variable code_my_closures : finite_map variable.

Local Abbreviation F := (rc_facts ctx).
Local Abbreviation T := (rc_tables ctx).

(* The metadata of a pattern variable: the [~var] that
   [rewrite_result_types] gives [patterns_for_unboxed_fields]. *)
Definition pattern_metadata (source : single_field_source)
    (use : FlambdaReaperUnboxing.field_use) : rewrite_metadata :=
  match source, use with
  | Field_no_source, _ => No_source
  | Field_one_source s, _ => Single_source s
  | Field_many_sources, FlambdaReaperUnboxing.Use_unknown =>
    Many_sources_any_usage
  | Field_many_sources, FlambdaReaperUnboxing.Use_known flow_to =>
    Many_sources_usages (get_direct_usages ctx flow_to)
  end.

(* [patterns_for_unboxed_fields] with no function slot, for the fields
   [u] of an allocation [block] whose uses are [uses]: the leaves of its
   pattern, in the order of the fields. [get_fields] gives the fields
   read; a field that the unboxing has and that is not read, which the
   OCaml stops at with a fatal error, has no variable, as have all where
   the fields read are of distinct subkinds. A field unboxed in turn is
   from its one source, with the uses of what it is read into; where it
   has not one source, or is read into names of which one has any
   usage (fatal errors), its leaves have no variable. *)
Fixpoint unboxed_leaves (kind : flambda_kind) (uses : list node)
    (block : node) (u : FlambdaReaperUnboxing.unboxed_field) {struct u}
    : list unboxed_leaf :=
  match u with
  | FlambdaReaperUnboxing.Not_unboxed _ => [([], kind, None)]
  | FlambdaReaperUnboxing.Unboxed fields =>
    let reads :=
      FlambdaReaperUnboxing.get_fields F T
        (FlambdaReaperUnboxing.add_usages_through_function_slots F T
           code_my_closures true uses)
    in
    let forget :=
      forgets_class
        (classify_field_map
           (filter (fun f => Keyed.is_bound field_eqb f reads)
              (map fst fields)))
    in
    (fix leaves
         (fields : list (field * FlambdaReaperUnboxing.unboxed_field))
         : list unboxed_leaf :=
       match fields with
       | [] => []
       | (f, u) :: fields =>
         match Keyed.lookup field_eqb f reads with
         | None => forgotten_leaves f u
         | Some use =>
           if forget then forgotten_leaves f u
           else
             let source := get_single_field_source ctx block f in
             match u with
             | FlambdaReaperUnboxing.Not_unboxed _ =>
               [([f], field_kind f, Some (pattern_metadata source use))]
             | FlambdaReaperUnboxing.Unboxed _ =>
               match use, source with
               | FlambdaReaperUnboxing.Use_known flow_to,
                 Field_one_source s =>
                 map (fun l => let '(p, k, m) := l in (f :: p, k, m))
                   (unboxed_leaves (field_kind f)
                      (get_direct_usages ctx flow_to) s u)
               | _, _ => forgotten_leaves f u
               end
             end
         end
         ++ leaves fields
       end) fields
  end.

(* The leaves of an unboxed variable of code, [code_var], from its
   allocation ([get_single_source]) and its uses ([get_usages]); with
   no allocation or no known use (fatal errors), none has a
   variable. *)
Definition unboxed_var_leaves (code_var : variable)
    (fields : FlambdaReaperUnboxing.unboxed_fields) : list unboxed_leaf :=
  let x := Node_variable code_var in
  match get_usages ctx x, get_single_source ctx x with
  | Ok usages, Ok allocation_point =>
    unboxed_leaves Value usages allocation_point
      (FlambdaReaperUnboxing.Unboxed fields)
  | _, _ =>
    map (fun pk => (fst pk, snd pk, None))
      (unboxed_field_paths Value (FlambdaReaperUnboxing.Unboxed fields))
  end.

(* [unknown_accessor] and [bottom_accessor] of traversals.ml. *)
Definition unknown_accessor (f : field) : flambda_type :=
  match f with
  | Field_is_int => any_naked_bool
  | _ => unknown (field_kind f)
  end.

Definition bottom_accessor (f : field) : flambda_type :=
  bottom (field_kind f).

(* [destructure_block_field_row_like_for_blocks] with no tag, and
   [destructure_block_field_row_like_block_case]. A map of tags is a
   singleton if it is a list of one binding. *)
Definition destructure_block_field (i : nat) (k : flambda_kind)
    (blocks : row_like_for_blocks) : flambda_type :=
  let of_case (case : row_like_block_case) :=
    match nth_error (case_maps_to case) i with
    | Some ty => ty
    | None => unknown k
    end
  in
  match blocks with
  | Row_like_for_blocks known_tags other_tags _ =>
    match other_tags with
    | Or_bottom.Ok case =>
      match known_tags with
      | [] => of_case case
      | _ :: _ => unknown k
      end
    | Or_bottom.Bottom =>
      match known_tags with
      | [(_, Or_unknown.Known case)] => of_case case
      | _ => unknown k
      end
    end
  end.

(* [destructure_head_of_kind_value_non_null] at the accessor of a field
   of the patterns of [patterns_for_unboxed_fields]: a field of a block
   ([Pattern.block], with no tag), [Is_int], [Get_tag], the contents of
   a boxed number ([Pattern.boxed_number]) and a value slot
   ([Pattern.closure]). A field of another sort is not destructured. *)
Definition destructure_non_null (f : field)
    (head : head_of_kind_value_non_null) : flambda_type :=
  match f, head with
  | Field_block _ _, Mutable_block _ => unknown_accessor f
  | Field_block i k, Variant _ _ _ blocks _ _ =>
    match blocks with
    | Or_unknown.Unknown => unknown_accessor f
    | Or_unknown.Known blocks => destructure_block_field i k blocks
    end
  | Field_is_int, (Mutable_block _ | Variant _ _ _ _ _ _) =>
    this_naked_immediate 0
  | Field_get_tag, Mutable_block _ => unknown_accessor f
  | Field_get_tag, Variant _ _ get_tag _ _ _ =>
    match get_tag with
    | None => unknown_accessor f
    | Some x => alias_type_of (Naked_number Kind_naked_immediate) (Var x)
    end
  | Field_boxed_number bn, Boxed_number k contents _ =>
    if kind_eqb (unboxed_kind bn) (unboxed_kind k) then contents
    else bottom_accessor f
  | Field_value_slot w, Closures closures _ =>
    match row_like_for_closures_get_env_var closures w with
    | Some (Or_unknown.Known ty) => ty
    | Some Or_unknown.Unknown | None => unknown_accessor f
    end
  | (Field_function_slot _ | Field_call_witness _ | Field_return_of_call _
     | Field_code_id_of_call_witness), _ =>
    unknown_accessor f
  | _, _ => bottom_accessor f
  end.

(* [destructure_expanded_head] at the accessor of a field. A head that
   is not of kind value (a fatal error) and a [None] of [expand_head]
   give the unknown type. *)
Definition destructure_field (f : field) (expanded : option Expanded_type.t)
    : flambda_type :=
  match expanded with
  | None => unknown_accessor f
  | Some expanded =>
    match Expanded_type.descr expanded with
    | Unknown => unknown_accessor f
    | Bottom => bottom_accessor f
    | Ok (Expanded_type.Head_value (Head_of_kind_value non_null _)) =>
      match non_null with
      | Unknown => unknown_accessor f
      | Bottom => bottom_accessor f
      | Ok head => destructure_non_null f head
      end
    | Ok _ => unknown_accessor f
    end
  end.

(* [fold_destructuring] along a path: the type of a leaf. *)
Fixpoint destructure_path (path : list field) (ty : flambda_type)
    : flambda_type :=
  match path with
  | [] => ty
  | f :: path =>
    destructure_path path (destructure_field f (expand_head tenv ty))
  end.

(* The metadata and the type to rewrite of each leaf of an unboxed
   variable of code [code_var], whose type in the result types is that of
   [type_var]: the type of the leaf in [fold_destructuring] of the alias
   to [type_var]; a leaf with no variable has the unknown type, under a
   metadata that forgets. *)
Definition unboxed_leaf_types (code_var type_var : variable)
    (fields : FlambdaReaperUnboxing.unboxed_fields)
    : list (rewrite_metadata * flambda_type) :=
  map
    (fun l =>
       let '(path, k, m) := l in
       match m with
       | None => (At_least_one_source_no_usages, unknown k)
       | Some m => (m, destructure_path path (result_alias tenv type_var))
       end)
    (unboxed_var_leaves code_var fields).

(* The types that [rewrite_result_types] gives the new parameters of an
   unboxed variable of code, one for each field of the unboxing, in the
   order of [Unboxed_fields.fold_with_kind]: the type of each leaf,
   rewritten under its metadata ([rewrite_arbitrary_type] in
   [rewrite_env_extension_with_extra_variables]), given the copies of
   the names it reaches. *)
Definition rewrite_unboxed_result_type (canon : canon_fn)
    (code_var type_var : variable)
    (fields : FlambdaReaperUnboxing.unboxed_fields) : list flambda_type :=
  map
    (fun p =>
       rewrite_arbitrary fx ctx tenv canon (rewrite_fuel tenv (snd p)) (fst p)
         (snd p))
    (unboxed_leaf_types code_var type_var fields).

(* [rewrite_result_types] with unboxing: [unboxed] has the unboxed fields
   of the names that the unboxing gives fields
   ([context.unboxing.unboxed_fields]). A parameter or a result whose
   variable of code is unboxed has a new variable for each field, with
   its type ([unboxed_leaf_types]), whatever the change of calling
   convention decides of it; the others are as in
   [rewrite_result_types]. The parameters start with the variables of
   the fields of [my_closure] if it is unboxed, which have no equation
   ([unbox_my_closure_vars]). *)
Definition rewrite_result_types_u
    (unboxed : list (node * FlambdaReaperUnboxing.unboxed_fields))
    (my_closure : variable)
    (params results : list (variable * keep_or_delete * variable))
    : rewritten_result_types :=
  let items (l : list (variable * keep_or_delete * variable)) :=
    flat_map
      (fun t =>
         let '(code_var, decision, type_var) := t in
         match Keyed.lookup node_eqb (Node_variable code_var) unboxed with
         | Some fields => unboxed_leaf_types code_var type_var fields
         | None =>
           match decision with
           | Keep =>
             [(metadata_for_name ctx (Node_variable code_var),
               result_alias tenv type_var)]
           | Delete => []
           end
         end)
      l
  in
  let param_items := items params in
  let result_items := items results in
  let numbered :=
    number_from (first_fresh_of_env tenv) (param_items ++ result_items)
  in
  let next :=
    Pos.of_nat (Pos.to_nat (first_fresh_of_env tenv) + length numbered)
  in
  let acc :=
    fold_left
      (fun acc t =>
         let '(_, (m, ty)) := t in
         fold_left get_canonical_with
           (reached_arbitrary fx ctx tenv (rewrite_fuel tenv ty) m ty) acc)
      numbered (Names_acc [] [] next)
  in
  let bindings :=
    map
      (fun t =>
         let '(new_var, (m, ty)) := t in
         (Name_var new_var,
          rewrite_arbitrary fx ctx tenv (canon_of acc) (rewrite_fuel tenv ty)
            m ty))
      numbered
  in
  let '(copies, _) := rewrite_in_depth fx ctx tenv (rounds_of_env tenv) acc []
  in
  let my_closure_vars :=
    match Keyed.lookup node_eqb (Node_variable my_closure) unboxed with
    | Some fields => map fst (FlambdaReaperUnboxing.fields_leaves fields)
    | None => []
    end
  in
  Rewritten_result_types
    (my_closure_vars
     ++ map fst (firstn (length param_items) numbered))
    (map fst (skipn (length param_items) numbered))
    (flat_map
       (fun p => match fst p with Name_var x => [x] | Name_symbol _ => [] end)
       copies)
    (bindings ++ copies).

End Unboxed.

(* ======================================================================== *)
(* Closures whose representation changes                                    *)
(* ======================================================================== *)

(* The closure case of [Rewriter.rewrite] where the one set that
   [refine_sets_of_closures] keeps changes its representation
   ([get_set_of_closures_changed_representation]: the representation
   [repr] of FlambdaReaperUnboxing.v that the changes of representation
   give its closures) and the closure cannot be external: the type of the
   closure in the new representation ([Expr.exactly_this_closure] at the
   new function slot of the closure), with
   - each function slot of the set at its new slot, of the code id that
     the type has for it if the closure there may be called, unknown if
     it is never called ([usages_of_function_slots_for_set_of_closures]);
   - the closure at each function slot of the set, at its new slot, of
     the type that the type has for it, rewritten under [Single_source]
     of that closure;
   - each new value slot of the representation, of the type at its path
     (a value slot, then the fields of what is unboxed in turn) in the
     type, rewritten under the metadata that [patterns_for_unboxed_fields]
     gives it with the patterns of the function slots, or unknown where
     it binds no variable.
   Where several sets are kept, or the closure may be external, the type
   is forgotten, as the closure case of [rewrite_rule] forgets it where a
   set changes ([rc_changed]). With [fix_null], the nullability is kept,
   as in [rewrite_closure]. The rest of the rewriting is [rewrite]'s
   ([rewrite_arbitrary_u]). None of it is extracted. *)

Section Changed.

Variable fx : fixes.
Variable ctx : rewrite_context.
Variable tenv : Typing_env.t.
Variable canon : canon_fn.
(* The relation [code_id_my_closure] (FlambdaReaperUnboxing.v,
   [bound_code_my_closures]). *)
Variable code_my_closures : finite_map variable.
(* The changes of representation of the names
   (FlambdaReaperUnboxing.unit_changed_representations). *)
Variable reprs : list (node * FlambdaReaperUnboxing.closure_repr).

Local Abbreviation F := (rc_facts ctx).
Local Abbreviation T := (rc_tables ctx).

(* [usages_of_function_slots_for_set_of_closures]: whether the closure
   may be called, with any arity or a known one. *)
Definition closure_called (closure : node) : bool :=
  match
    get_one_field_usage_of_constructors ctx [closure]
      (Field_call_witness Unknown_arity_code_pointer)
  with
  | Unknown | Ok _ => true
  | Bottom =>
    match
      get_one_field_usage_of_constructors ctx [closure]
        (Field_call_witness Known_arity_code_pointer)
    with
    | Unknown | Ok _ => true
    | Bottom => false
    end
  end.

(* [patterns_for_unboxed_fields] with the patterns of the function slots:
   the leaves of the fields [fields] of the representation, at the
   closure [source] of the type, the closures of the set being read at
   [reads] ([get_fields_usage_of_constructors]), in the order of the
   fields; as [unboxed_leaves] at a level, and [unboxed_leaves] for a
   value slot unboxed in turn. With no field read, the pattern is
   [Pattern.closure] of the function slots alone, and binds no value
   slot. *)
Definition changed_leaves (source : node)
    (reads : list (field * FlambdaReaperUnboxing.field_use))
    (fields : FlambdaReaperUnboxing.unboxed_fields) : list unboxed_leaf :=
  let cls :=
    classify_field_map
      (filter (fun f => Keyed.is_bound field_eqb f reads) (map fst fields))
  in
  match cls with
  | Class_empty => []
  | _ =>
    flat_map
      (fun p : field * FlambdaReaperUnboxing.unboxed_field =>
         let '(f, u) := p in
         match Keyed.lookup field_eqb f reads with
         | None => forgotten_leaves f u
         | Some use =>
           if forgets_class cls then forgotten_leaves f u
           else
             let source_f := get_single_field_source ctx source f in
             match u with
             | FlambdaReaperUnboxing.Not_unboxed _ =>
               [([f], field_kind f, Some (pattern_metadata ctx source_f use))]
             | FlambdaReaperUnboxing.Unboxed _ =>
               match use, source_f with
               | FlambdaReaperUnboxing.Use_known flow_to,
                 Field_one_source s =>
                 map (fun l => let '(p, k, m) := l in (f :: p, k, m))
                   (unboxed_leaves ctx code_my_closures (field_kind f)
                      (get_direct_usages ctx flow_to) s u)
               | _, _ => forgotten_leaves f u
               end
             end
         end)
      fields
  end.

(* The case of [Rewriter.rewrite] of a closure type whose one set kept
   changes, with the closure local: its representation, the set, the
   slot and allocation mode of the closure, and the entry of the type. *)
Definition changed_rule (usages : rewrite_metadata) (ty : flambda_type)
    : option
        (FlambdaReaperUnboxing.closure_repr * set_of_closures * function_slot
         * Alloc_mode.t * closures_entry) :=
  if forgets usages then None
  else
    match kind ty with
    | Value =>
      match Provers.meet_single_closures_entry tenv ty with
      | Some (Meet_shortcut.Known_result (slot, alloc_mode, entry, _)) =>
        match entry with
        | Closures_entry _ closure_types value_slot_types =>
          let code_ids :=
            map (fun g => (g, entry_code_id entry g)) (domain closure_types)
          in
          let mbl :=
            if negb (is_nil (domain value_slot_types))
               || negb (is_nil (domain closure_types))
               || existsb
                    (fun p =>
                       match snd p with
                       | Or_unknown.Known _ => true
                       | Or_unknown.Unknown => false
                       end)
                    code_ids
            then Must_be_local
            else Can_be_external
          in
          match refine_sets_of_closures fx ctx usages slot code_ids mbl with
          | ([set], Must_be_local) =>
            match Keyed.lookup node_eqb (closure_at set slot) reprs with
            | Some repr => Some (repr, set, slot, alloc_mode, entry)
            | None => None
            end
          | _ => None
          end
        end
      | _ => None
      end
    | _ => None
    end.

(* The type of the closure in the new representation. *)
Definition rewrite_changed
    (rw : rewrite_metadata -> flambda_type -> flambda_type)
    (repr : FlambdaReaperUnboxing.closure_repr) (set : set_of_closures)
    (slot : function_slot) (alloc_mode : Alloc_mode.t)
    (entry : closures_entry) (ty : flambda_type) : flambda_type :=
  let expanded := expand_head tenv ty in
  let new_slot := FlambdaReaperUnboxing.rename_function_slot repr in
  let reads :=
    FlambdaReaperUnboxing.get_fields_of_constructors F T (rc_is_local ctx)
      (map (fun p => fst (snd p)) set)
  in
  let leaves :=
    changed_leaves (closure_at set slot) reads
      (FlambdaReaperUnboxing.cr_fields repr)
  in
  let ty' :=
    exactly_this_closure (new_slot slot)
      (map
         (fun p : function_slot * (node * Or_unknown.t code_id) =>
            (new_slot (fst p),
             if closure_called (fst (snd p))
             then
               match entry_code_id entry (fst p) with
               | Or_unknown.Known code => Or_unknown.Known (Function_type code)
               | Or_unknown.Unknown => Or_unknown.Unknown
               end
             else Or_unknown.Unknown))
         set)
      (map
         (fun p : function_slot * (node * Or_unknown.t code_id) =>
            (new_slot (fst p),
             rw (Single_source (fst (snd p)))
               (destructure_closure
                  (fun c => row_like_for_closures_get_closure c (fst p))
                  expanded)))
         set)
      (map
         (fun q : unboxed_leaf * (variable * flambda_kind) =>
            let '((path, k, m), (slot', _)) := q in
            (slot',
             match m with
             | None => unknown k
             | Some m => rw m (destructure_path tenv path ty)
             end))
         (combine leaves
            (FlambdaReaperUnboxing.fields_leaves
               (FlambdaReaperUnboxing.cr_fields repr))))
      alloc_mode
  in
  if fix_null fx && expanded_maybe_null tenv ty
  then
    match ty' with
    | Type_value (Ok (No_alias (Head_of_kind_value non_null _))) =>
      Type_value (Ok (No_alias (Head_of_kind_value non_null (Maybe_null None))))
    | _ => ty'
    end
  else ty'.

(* [rewrite_arbitrary_type] and [rewrite] with the changes of
   representation: [rewrite_arbitrary] and [rewrite], but where a set
   kept changes. *)
Fixpoint rewrite_arbitrary_u (fuel : nat) (m : rewrite_metadata)
    (ty : flambda_type) {struct fuel} : flambda_type :=
  match fuel with
  | O => unknown (kind ty)
  | S fuel =>
    match get_alias_opt ty with
    | Some alias =>
      match
        Typing_env.get_canonical_simple_exn tenv (Some Name_mode.in_types)
          None alias
      with
      | Some (Some (Const c)) => rewrite_u fuel m (type_for_const c)
      | Some (Some (Var x)) =>
        if fix_alias fx && forgets m then unknown (kind ty)
        else
          alias_type_of (kind ty)
            (simple_of_name (canon (Name_var x) (kind ty) m))
      | Some (Some (Symbol s)) =>
        if fix_alias fx && forgets m then unknown (kind ty)
        else
          alias_type_of (kind ty)
            (simple_of_name (canon (Name_symbol s) (kind ty) m))
      | Some None | None => unknown (kind ty)
      end
    | None => rewrite_u fuel m ty
    end
  end

with rewrite_u (fuel : nat) (m : rewrite_metadata) (ty : flambda_type)
    {struct fuel} : flambda_type :=
  match fuel with
  | O => unknown (kind ty)
  | S fuel =>
    match changed_rule m ty with
    | Some (repr, set, slot, alloc_mode, entry) =>
      rewrite_changed (rewrite_arbitrary_u fuel) repr set slot alloc_mode
        entry ty
    | None =>
      match rewrite_rule fx ctx tenv m ty with
      | Rule_forget => unknown (kind ty)
      | Rule_identity =>
        rewrite_identity fx ctx tenv (rewrite_arbitrary_u fuel) m ty
      | Rule_closure spec =>
        rewrite_closure fx tenv (rewrite_arbitrary_u fuel) spec ty
      end
    end
  end.

End Changed.
