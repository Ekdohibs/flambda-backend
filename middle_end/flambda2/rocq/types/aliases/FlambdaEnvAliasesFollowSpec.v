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

(* Following aliases through a term, with the aliases of the typing
   environment: a normalisation that is an equivalence.

   This file is to be reviewed: what is known where a subterm is
   ([Known.t]) and the functions that keep it ([Known.add_equal],
   [Known.forget], [Known.bind], [Known.under_let],
   [Known.under_let_static]); [follow_aliases] and what it is made of;
   what it is for a static environment and an environment of the
   machine to agree with what is known ([known_respected]); and the
   statements.
   [follow_aliases] is written to be extracted and run by the validation
   of middle_end/flambda2/tests/semantics/, in the place of
   FlambdaAliasesSpec.follow_aliases, which is written with a list of
   variables and types. This one is written with the aliases of
   FlambdaEnvAliases.v (after middle_end/flambda2/types/env/aliases.ml),
   and knows the closures of a set of several functions, and the
   closures that a [Let_static] binds symbols to.

   What is known where a subterm is ([Known.t]):
   - the aliases: classes of simples that are equal, each with its
     canonical element ([Aliases.t]); with them, when each variable was
     bound ([Known.binding_times], [Known.next_binding_time]), which is
     what [Aliases.add] chooses the canonical element by;
   - the closures: for some variables, the closure that the variable was
     bound to by a [Let] ([Known.closure]): the code of a closure with
     no function slot named, or the function slot, the functions and the
     value slots of a closure of a set;
   - the closures of symbols: for some symbols, the closure that a
     [Let_static] bound the symbol to ([Known.static_closure]): the
     same, and for a closure of a set the symbols that the set names for
     its function slots.

   [follow_aliases] goes through a term and
   - replaces every [Simple] by the canonical element of its class
     ([Known.canonical_simple]), as Simplify does
     (simplify/simplify_simple.ml);
   - makes a call of a variable or of a symbol that is known to be a
     closure a direct call of its code ([canonical_call_kind]);
   - replaces the projection of a value slot from such a variable or
     symbol by what the slot holds ([canonical_project], and
     [canonical_project_from] where the projection names the function
     slot of the closure: only if it is the slot that the closure is
     at);
   - replaces the projection of a function slot from a symbol known to
     be the closure of a static set at the slot that the projection
     names by the symbol that the set names for the other slot
     ([canonical_project_function_slot]).
   There is no substitution, and a [Let] stays where it is, to be
   removed if it is then unused (FlambdaNormaliseSpec.v).

   Going under a [Let] ([Known.under_let]) the variable is bound
   ([Known.bind]), and then:
   - if it is bound to a [Simple], the two are made aliases
     ([Known.add_equal], with [Aliases.add] on their canonical
     elements). The variable has the latest binding time, so it is the
     one demoted, and its uses become the canonical element of the
     [Simple];
   - if it is bound to a closure, that is recorded;
   - if it is bound to the projection of a function slot from a variable
     known to be a closure of a set at the slot that the projection
     names, it is recorded to be the closure of that set at the other
     slot. This is how the closures of a set of several functions are
     all known: the compiler's [Let] of a set is a [Let] of the first
     closure and one of each other closure as its projection from the
     first (FlambdaModel.let_closures).
   Going under a [Let_static] ([Known.under_let_static]) each symbol
   that the group binds to a closure is recorded with it. What is
   recorded of a symbol stays true as the term runs on: the machine
   does not bind a symbol twice (that is undefined:
   FlambdaSemanticsSpec.eval_static_bindings), and a static environment
   only grows ([known_respected_extend_statement]). The value slots of
   a static closure are [Simple]s, read where the group is.
   Handlers of non-recursive continuations are gone into, with what is
   known where they are defined and their parameters bound; those of
   recursive ones and code are left as they are.

   Binders. In the model a binder may bind a variable that is in scope
   already, and then what was known of the variable is about another
   one. So binding a variable first forgets it ([Known.forget]):
   - of the aliases, where the variable is written in them: the
     structure is made again from the names that it has demoted, each
     with its canonical element, leaving the variable out. If the
     variable was the canonical element of a class, the other names of
     the class are made aliases of one another;
   - of the closures: the one recorded for the variable, and those, of
     variables and of symbols, with a value slot that holds it.
   Where the variable is not written in the aliases they are left as
   they are, which is the case of every binder of a term whose bound
   variables are all different. The compiler has no such operation, and
   needs none: it does not bind a variable twice. The validator applies
   [follow_aliases] to terms that its own rewrites have made, by
   inlining for one, in which a variable may be bound twice.

   What is not as in the compiler:
   - Simplify keeps a typing environment, of which the aliases are a
     part, and learns what a closure is from the type of the variable.
     Here the closures known are two lists of their own, a stand-in
     until the types of closures and the provers are used for it.
   - A symbol is known to be bound to a closure only in the body of the
     [Let_static] that binds it, and in the handlers defined there: not
     in code, and not where the term has not gone through the binding.
     Simplify has the type of a symbol wherever the symbol is used.
   - The projection of a function slot from a variable known to be a
     closure is left as it is: the variable that holds the closure at
     the other slot, if there is one, is not looked for.
   - A variable that is not bound in the term (a parameter of the code
     that the term is the body of, say) has the binding time of the
     variables of other compilation units, so that it is earlier than
     every variable that the term binds.
   - Name modes are not read: every name that the aliases answer is
     taken to be usable in a term
     ([Aliases.get_canonical_ignoring_name_mode]). What is known here
     has no variable that exists only in types.
   - Where [Aliases.add] has no answer (a fatal error in the compiler),
     or answers bottom, the aliases are left as they were.

   The statements:
   - every static environment and environment agree with knowing
     nothing, and what is known stays true when the static environment
     binds more;
   - in an environment that agrees with what is known, the canonical
     element of a [Simple] has the value of the [Simple];
   - an environment that agrees with what is known still does when a
     variable is forgotten and then bound to any value, and when the
     variable of a [Let] is bound to the value of what it is bound to;
     and the static environment that a [Let_static] makes agrees with
     what is known in its body;
   - two simples that were known to be equal, neither of them the
     variable forgotten, are still known to be equal, where the aliases
     are consistent; and they stay consistent under binders;
   - the result is equivalent to the term in every static environment
     and environment that agree with what is known
     ([follow_aliases_respected_statement]), so, starting from nothing
     known, in every environment
     ([follow_aliases_equivalent_statement]): it is not only a
     refinement, and may be applied on both sides of a comparison;
   - it commutes with renaming ([rename_follow_aliases_statement]):
     the result for a renamed term, with what is known renamed
     ([rename_known]), is the result renamed. There is no premise: the
     names of the term need not be different, as a binder forgets.

   The proofs are in FlambdaEnvAliasesFollowProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaNominalSpec FlambdaNormaliseSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesRenamingSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known where a subterm is                                         *)
(* ======================================================================== *)

Module Known.

(* The closure that a variable is known to be: one with no function slot
   named ([Set_of_closures]), with its code; or the closure at [slot] of
   a set ([Closure_of_set]), with the functions of the set. Either has
   the [Simple]s that its value slots were given. *)
Inductive closure : Type :=
  | Of_code (code : code_id) (value_slots : list (value_slot * simple))
  | Of_set (slot : function_slot)
      (function_decls : list (function_slot * code_id))
      (value_slots : list (value_slot * simple)).

Definition closure_value_slots (c : closure) : list (value_slot * simple) :=
  match c with
  | Of_code _ value_slots | Of_set _ _ value_slots => value_slots
  end.

(* The code that a call of the closure runs: that of the function slot
   it is at. *)
Definition closure_code (c : closure) : option code_id :=
  match c with
  | Of_code code _ => Some code
  | Of_set slot function_decls _ => lookup slot function_decls
  end.

(* The function slot that the closure is at. *)
Definition closure_slot (c : closure) : function_slot :=
  match c with
  | Of_code _ _ => unnamed_function_slot
  | Of_set slot _ _ => slot
  end.

(* The closure that a symbol is known to be bound to: the closure, and
   the symbols that its set names for its function slots (none for a
   closure with no function slot named). *)
Record static_closure : Type := make_static_closure {
  as_closure : closure;
  closure_symbols : list (function_slot * symbol)
}.

(* What is known. [binding_times] has the variables that have been bound
   on the way to the subterm, each with the time of its last binding. *)
Record t : Type := make {
  aliases : Aliases.t;
  binding_times : Keyed.finite_map name (unit * Binding_time.With_name_mode.t);
  next_binding_time : Binding_time.t;
  closures : list (variable * closure);
  static_closures : list (symbol * static_closure)
}.

Definition empty : t :=
  make Aliases.empty [] Binding_time.earliest_var [] [].

(* The binding time of a name that [binding_times] has not: a symbol has
   that of symbols; a variable is not bound in the term, and has that of
   the variables of other compilation units. *)
Definition binding_time_resolver (n : name) : Binding_time.With_name_mode.t :=
  match n with
  | Name_var _ => Binding_time.With_name_mode.imported_variables
  | Name_symbol _ => Binding_time.With_name_mode.symbols
  end.

(* The canonical element of the class of a simple. *)
Definition canonical_simple (known : t) (s : simple) : simple :=
  canonical_element_of (aliases known) s.

(* Two simples have been found to be equal: their classes are made one,
   by [Aliases.add] on their canonical elements. *)
Definition add_equal
    (binding_times
      : Keyed.finite_map name (unit * Binding_time.With_name_mode.t))
    (aliases : Aliases.t) (e e' : simple) : Aliases.t :=
  let canonical_element1 := canonical_element_of aliases e in
  let canonical_element2 := canonical_element_of aliases e' in
  if simple_equal canonical_element1 canonical_element2
  then aliases
  else
    match
      Aliases.add binding_time_resolver binding_times aliases
        canonical_element1 canonical_element2
    with
    | Some (Or_bottom.Ok result) => Aliases.add_result_t result
    | Some Or_bottom.Bottom | None => aliases
    end.

(* The aliases without the variable [x]: the structure made again from
   the names that it has demoted, but [x], each with its canonical
   element; where that is [x], with the first of the names demoted to
   [x] in its place. The compiler has no such function. *)
Definition forget_in_aliases
    (binding_times
      : Keyed.finite_map name (unit * Binding_time.With_name_mode.t))
    (aliases : Aliases.t) (x : variable) : Aliases.t :=
  if negb (among x (aliases_free_variables aliases))
  then aliases
  else
    let demoted_to_x n :=
      simple_equal (Aliases.get_canonical_ignoring_name_mode aliases n)
        (Var x)
    in
    let demoted :=
      filter (fun n => negb (name_eqb n (Name_var x)))
        (Keyed.domain name_eqb (Aliases.canonical_elements aliases))
    in
    let in_place_of_x := find demoted_to_x demoted in
    List.fold_left
      (fun rebuilt n =>
         if demoted_to_x n
         then
           match in_place_of_x with
           | Some n' =>
             add_equal binding_times rebuilt (simple_of_name n)
               (simple_of_name n')
           | None => rebuilt
           end
         else
           add_equal binding_times rebuilt (simple_of_name n)
             (Aliases.get_canonical_ignoring_name_mode aliases n))
      demoted Aliases.empty.

(* What is known that is not about [x]. *)
Definition forget (known : t) (x : variable) : t :=
  make (forget_in_aliases (binding_times known) (aliases known) x)
    (binding_times known) (next_binding_time known)
    (filter
       (fun entry =>
          negb (Pos.eqb x (fst entry))
          && negb (var_in_value_slots x (closure_value_slots (snd entry))))
       (closures known))
    (filter
       (fun entry =>
          negb
            (var_in_value_slots x
               (closure_value_slots (as_closure (snd entry)))))
       (static_closures known)).

(* [x] is bound: nothing is known of it but when it was bound. *)
Definition bind (known : t) (x : variable) : t :=
  let known := forget known x in
  make (aliases known)
    ((Name_var x,
      (tt,
       Binding_time.With_name_mode.create (next_binding_time known)
         Name_mode.normal))
     :: binding_times known)
    (Binding_time.succ (next_binding_time known)) (closures known)
    (static_closures known).

(* The parameters of a handler are bound, in their order. *)
Definition bind_all (known : t) (xs : list variable) : t :=
  List.fold_left bind xs known.

Definition with_aliases (known : t) (aliases : Aliases.t) : t :=
  make aliases (binding_times known) (next_binding_time known)
    (closures known) (static_closures known).

Definition add_closure (known : t) (x : variable) (c : closure) : t :=
  make (aliases known) (binding_times known) (next_binding_time known)
    ((x, c) :: closures known) (static_closures known).

(* The closure that a simple is known to be: the one recorded for a
   variable, or the one that a symbol is known to be bound to. *)
Definition find_closure (known : t) (s : simple) : option closure :=
  match s with
  | Var x => lookup x (closures known)
  | Symbol sym => option_map as_closure (lookup sym (static_closures known))
  | Const _ => None
  end.

(* What is known in the body of a [Let] of [x] to [d]: [x] is bound;
   then it is an alias of [d] if that is a [Simple], or the closure that
   [d] makes, or the closure of the same set at another slot if [d] is
   the projection of a function slot from a variable that is known to
   be a closure. A [Simple] or a value slot that is [x] is the variable
   that [x] was before the [Let], of which nothing is known any more.
   (The projection of a function slot from a symbol is a symbol, which
   [canonical_project_function_slot] below puts in its place where it
   is known; nothing is recorded for it here.) *)
Definition under_let (known : t) (x : variable) (d : named) : t :=
  let known := bind known x in
  match d with
  | Simple s =>
    if var_in_simple x s
    then known
    else
      with_aliases known
        (add_equal (binding_times known) (aliases known) (Var x) s)
  | Set_of_closures code value_slots =>
    if var_in_value_slots x value_slots
    then known
    else add_closure known x (Of_code code value_slots)
  | Closure_of_set slot function_decls value_slots =>
    if var_in_value_slots x value_slots
    then known
    else add_closure known x (Of_set slot function_decls value_slots)
  | Project_function_slot move_from move_to closure =>
    match closure with
    | Var y =>
      match lookup y (closures known) with
      | Some (Of_set slot function_decls value_slots) =>
        if Pos.eqb slot move_from
        then add_closure known x (Of_set move_to function_decls value_slots)
        else known
      | Some (Of_code _ _) | None => known
      end
    | Symbol _ | Const _ => known
    end
  | Prim _ _ | Project_value_slot _ _ | Project_value_slot_from _ _ _ =>
    known
  end.

(* The closure that a static binding binds its symbol to, if it binds a
   symbol to a closure. *)
Definition static_closure_of_binding (binding : static_binding)
    : option (symbol * static_closure) :=
  match binding with
  | Bind_symbol s (Static_const_set_of_closures code value_slots) =>
    Some (s, make_static_closure (Of_code code value_slots) [])
  | Bind_symbol s
      (Static_const_closure_of_set slot closure_symbols function_decls
         value_slots) =>
    Some
      (s,
       make_static_closure (Of_set slot function_decls value_slots)
         closure_symbols)
  | Bind_symbol _
      (Static_const_block _ _ | Static_const_boxed_number _
      | Static_const_immutable_string _
      | Static_const_immutable_value_array _)
  | Bind_code _ _ | Bind_deleted_code _ =>
    None
  end.

(* What is known in the body of a [Let_static]: each symbol that the
   group binds to a closure is bound to that closure. *)
Definition under_let_static (known : t) (bindings : list static_binding)
    : t :=
  make (aliases known) (binding_times known) (next_binding_time known)
    (closures known)
    (flat_map
       (fun binding =>
          match static_closure_of_binding binding with
          | Some entry => [entry]
          | None => []
          end)
       bindings
     ++ static_closures known).

End Known.

(* ======================================================================== *)
(* Following aliases                                                        *)
(* ======================================================================== *)

Definition canonical_value_slots (known : Known.t)
    (slots : list (value_slot * simple)) : list (value_slot * simple) :=
  map (fun slot => (fst slot, Known.canonical_simple known (snd slot))) slots.

(* The projection of a value slot from a known closure is what the slot
   holds. *)
Definition canonical_project (known : Known.t) (slot : value_slot)
    (closure : simple) : named :=
  match Known.find_closure known closure with
  | Some c =>
    match lookup slot (Known.closure_value_slots c) with
    | Some contents => Simple contents
    | None => Project_value_slot slot closure
    end
  | None => Project_value_slot slot closure
  end.

(* The same where the projection names the function slot of the closure
   it projects from: only if the closure is known to be at that slot. *)
Definition canonical_project_from (known : Known.t)
    (project_from : function_slot) (slot : value_slot) (closure : simple)
    : named :=
  match Known.find_closure known closure with
  | Some c =>
    if Pos.eqb (Known.closure_slot c) project_from
    then
      match lookup slot (Known.closure_value_slots c) with
      | Some contents => Simple contents
      | None => Project_value_slot_from project_from slot closure
      end
    else Project_value_slot_from project_from slot closure
  | None => Project_value_slot_from project_from slot closure
  end.

(* The projection of a function slot from a symbol known to be bound to
   the closure of a static set, at the slot that the projection names:
   the symbol that the set names for the other slot, if the set has that
   slot. *)
Definition canonical_project_function_slot (known : Known.t)
    (move_from move_to : function_slot) (closure : simple) : named :=
  match closure with
  | Symbol s =>
    match lookup s (Known.static_closures known) with
    | Some c =>
      match Known.as_closure c with
      | Known.Of_set slot function_decls _ =>
        if Pos.eqb slot move_from && is_bound move_to function_decls
        then
          match lookup move_to (Known.closure_symbols c) with
          | Some s' => Simple (Symbol s')
          | None => Project_function_slot move_from move_to closure
          end
        else Project_function_slot move_from move_to closure
      | Known.Of_code _ _ => Project_function_slot move_from move_to closure
      end
    | None => Project_function_slot move_from move_to closure
    end
  | Var _ | Const _ => Project_function_slot move_from move_to closure
  end.

Definition canonical_named (known : Known.t) (d : named) : named :=
  match d with
  | Simple s => Simple (Known.canonical_simple known s)
  | Prim p args => Prim p (map (Known.canonical_simple known) args)
  | Set_of_closures code slots =>
    Set_of_closures code (canonical_value_slots known slots)
  | Closure_of_set slot decls slots =>
    Closure_of_set slot decls (canonical_value_slots known slots)
  | Project_value_slot slot closure =>
    canonical_project known slot (Known.canonical_simple known closure)
  | Project_value_slot_from project_from slot closure =>
    canonical_project_from known project_from slot
      (Known.canonical_simple known closure)
  | Project_function_slot move_from move_to closure =>
    canonical_project_function_slot known move_from move_to
      (Known.canonical_simple known closure)
  end.

Definition canonical_static_const (known : Known.t) (const : static_const)
    : static_const :=
  match const with
  | Static_const_set_of_closures code slots =>
    Static_const_set_of_closures code (canonical_value_slots known slots)
  | Static_const_closure_of_set slot symbols decls slots =>
    Static_const_closure_of_set slot symbols decls
      (canonical_value_slots known slots)
  | Static_const_block tag fields =>
    Static_const_block tag (map (Known.canonical_simple known) fields)
  | Static_const_boxed_number contents =>
    Static_const_boxed_number (Known.canonical_simple known contents)
  | Static_const_immutable_string bytes => Static_const_immutable_string bytes
  | Static_const_immutable_value_array elements =>
    Static_const_immutable_value_array
      (map (Known.canonical_simple known) elements)
  end.

Definition canonical_static_binding (known : Known.t)
    (binding : static_binding) : static_binding :=
  match binding with
  | Bind_code id c => Bind_code id c
  | Bind_deleted_code id => Bind_deleted_code id
  | Bind_symbol s const => Bind_symbol s (canonical_static_const known const)
  end.

Definition canonical_apply_cont (known : Known.t) (ac : apply_cont)
    : apply_cont :=
  Apply_cont_expr (ac_continuation ac)
    (map (Known.canonical_simple known) (ac_args ac)) (ac_trap_action ac).

(* A call of unknown arity of a known closure is a direct call of its
   code. A call of known arity is left: it is undefined wherever the
   code of its callee has not as many parameters as the call has groups
   of arguments
   (FlambdaKnownAritySpec.known_arity_other_arity_statement), and
   nothing here says that it is the direct call. *)
Definition canonical_call_kind (known : Known.t) (callee : simple)
    (kind : call_kind) : call_kind :=
  match kind with
  | Indirect_unknown_arity =>
    match Known.find_closure known callee with
    | Some c =>
      match Known.closure_code c with
      | Some code => Direct code
      | None => Indirect_unknown_arity
      end
    | None => Indirect_unknown_arity
    end
  | Direct _ | Indirect_known_arity _ => kind
  end.

Definition canonical_apply (known : Known.t) (ap : apply_expr) : apply_expr :=
  let callee := Known.canonical_simple known (ap_callee ap) in
  Apply_expr callee (canonical_call_kind known callee (ap_call_kind ap))
    (map (Known.canonical_simple known) (ap_args ap)) (ap_continuation ap)
    (ap_exn_continuation ap)
    (map (fun arg => (Known.canonical_simple known (fst arg), snd arg))
       (ap_exn_extra_args ap))
    (ap_args_arity ap) (ap_return_arity ap).

Fixpoint follow_aliases (known : Known.t) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    let d' := canonical_named known d in
    Let x kind d' (follow_aliases (Known.under_let known x d') body)
  | Let_static bindings body =>
    let bindings' := map (canonical_static_binding known) bindings in
    Let_static bindings'
      (follow_aliases (Known.under_let_static known bindings') body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (follow_aliases_in_handler known h)
      (follow_aliases known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (follow_aliases known body)
  | Apply ap => Apply (canonical_apply known ap)
  | Apply_cont ac => Apply_cont (canonical_apply_cont known ac)
  | Switch scrutinee arms =>
    Switch (Known.canonical_simple known scrutinee)
      (map (fun arm => (fst arm, canonical_apply_cont known (snd arm))) arms)
  | Invalid => Invalid
  end
with follow_aliases_in_handler (known : Known.t) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (follow_aliases (Known.bind_all known params) handler)
  end.

(* ======================================================================== *)
(* What is known, and the environments of the machine                       *)
(* ======================================================================== *)

(* The value of a closure whose value slots have the values
   [slot_values]. *)
Definition closure_value (c : Known.closure)
    (slot_values : list (value_slot * value)) : value :=
  match c with
  | Known.Of_code code _ => Value_closure code slot_values
  | Known.Of_set slot function_decls _ =>
    Value_closure_of_set slot function_decls slot_values
  end.

(* The variable [x] is the closure [c]: its value is that closure, with
   the values that the [Simple]s of the value slots have; and the
   closure has code, which is to say, for a closure of a set, that its
   function slot is one of the set's (a value of that shape at another
   slot is not a closure: FlambdaValueSpec.closure_in_set). *)
Definition closure_respected (vars : env) (x : variable) (c : Known.closure)
    : Prop :=
  Known.closure_code c <> None
  ∧ ∃ slot_values : list (value_slot * value),
      eval_value_slots vars (Known.closure_value_slots c) = Some slot_values
      ∧ lookup x vars = Some (closure_value c slot_values).

(* What a symbol is bound to when it is bound to the closure [c] whose
   value slots have the values [slot_values]. *)
Definition static_closure_value (c : Known.static_closure)
    (slot_values : list (value_slot * value)) : statically_allocated_value :=
  match Known.as_closure c with
  | Known.Of_code code _ => Static_closure code slot_values
  | Known.Of_set slot function_decls _ =>
    Static_closure_of_set slot (Known.closure_symbols c) function_decls
      slot_values
  end.

(* The symbol [s] is bound to the closure [c]: the static environment
   binds it to that closure, with the values that the [Simple]s of the
   value slots have; and the closure has code, as for a variable. *)
Definition static_closure_respected (statics : static_env) (vars : env)
    (s : symbol) (c : Known.static_closure) : Prop :=
  Known.closure_code (Known.as_closure c) <> None
  ∧ ∃ slot_values : list (value_slot * value),
      eval_value_slots vars (Known.closure_value_slots (Known.as_closure c))
      = Some slot_values
      ∧ lookup s (static_values statics)
        = Some (static_closure_value c slot_values).

(* A static environment and an environment of the machine agree with
   what is known: the environment respects the aliases
   (FlambdaEnvAliasesSpec.v), every closure recorded for a variable is
   what the variable is, and every closure recorded for a symbol is what
   the static environment binds the symbol to. *)
Definition known_respected (known : Known.t) (statics : static_env)
    (vars : env) : Prop :=
  aliases_respected (Known.aliases known) vars
  ∧ (∀ (x : variable) (c : Known.closure),
      (x, c) ∈ Known.closures known -> closure_respected vars x c)
  ∧ ∀ (s : symbol) (c : Known.static_closure),
      (s, c) ∈ Known.static_closures known ->
      static_closure_respected statics vars s c.

Definition empty_known_respected_statement : Prop :=
  ∀ (statics : static_env) (vars : env),
    known_respected Known.empty statics vars.

(* What is known stays true when the static environment binds more. *)
Definition known_respected_extend_statement : Prop :=
  ∀ (known : Known.t) (statics statics' : static_env) (vars : env),
    known_respected known statics vars ->
    statics_extend statics statics' ->
    known_respected known statics' vars.

(* The canonical element of a [Simple] has its value. *)
Definition canonical_simple_respected_statement : Prop :=
  ∀ (known : Known.t) (statics : static_env) (vars : env) (s : simple),
    known_respected known statics vars ->
    eval_simple vars (Known.canonical_simple known s) = eval_simple vars s.

(* What is left when a variable is forgotten is about the other
   variables only: it holds whatever the variable is then bound to. *)
Definition forget_respected_statement : Prop :=
  ∀ (known : Known.t) (statics : static_env) (vars : env) (x : variable)
    (v : value),
    known_respected known statics vars ->
    known_respected (Known.forget known x) statics ((x, v) :: vars).

Definition bind_respected_statement : Prop :=
  ∀ (known : Known.t) (statics : static_env) (vars : env) (x : variable)
    (v : value),
    known_respected known statics vars ->
    known_respected (Known.bind known x) statics ((x, v) :: vars).

(* Forgetting a variable loses nothing of the aliases that is not about
   it: two simples that were known to be equal, neither of them the
   variable, still are. The structure has to be consistent
   (FlambdaEnvAliasesSpec.aliases_consistent), which the one that
   [follow_aliases] keeps is (the two statements after this one). *)
Definition forget_aliased_statement : Prop :=
  ∀ (binding_times
      : Keyed.finite_map name (unit * Binding_time.With_name_mode.t))
    (aliases : Aliases.t) (x : variable) (e e' : simple),
    aliases_consistent aliases ->
    aliased aliases e e' ->
    e <> Var x ->
    e' <> Var x ->
    aliased (Known.forget_in_aliases binding_times aliases x) e e'.

(* The aliases stay consistent, from knowing nothing on. *)
Definition bind_consistent_statement : Prop :=
  ∀ (known : Known.t) (x : variable),
    aliases_consistent (Known.aliases known) ->
    aliases_consistent (Known.aliases (Known.bind known x)).

Definition under_let_consistent_statement : Prop :=
  ∀ (known : Known.t) (x : variable) (d : named),
    aliases_consistent (Known.aliases known) ->
    aliases_consistent (Known.aliases (Known.under_let known x d)).

(* Under a [Let]: the environment in which the body runs agrees with
   what is known there. *)
Definition under_let_respected_statement : Prop :=
  ∀ (known : Known.t) (statics : static_env) (vars : env) (x : variable)
    (d : named) (v : value),
    known_respected known statics vars ->
    eval_named statics vars d = Some v ->
    known_respected (Known.under_let known x d) statics ((x, v) :: vars).

(* Under a [Let_static]: the static environment that the group makes
   agrees, with the environment, with what is known in the body. *)
Definition under_let_static_respected_statement : Prop :=
  ∀ (known : Known.t) (statics extended : static_env) (vars : env)
    (bindings : list static_binding),
    known_respected known statics vars ->
    eval_static_bindings vars bindings statics = Some extended ->
    known_respected (Known.under_let_static known bindings) extended vars.

(* ======================================================================== *)
(* The result is equivalent to the term                                     *)
(* ======================================================================== *)

Definition follow_aliases_respected_statement : Prop :=
  ∀ (known : Known.t) (e : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    known_respected known statics vars ->
    equivalent (State e vars conts statics hp rd)
      (State (follow_aliases known e) vars conts statics hp rd).

Definition follow_aliases_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (follow_aliases Known.empty e).

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* What is known, renamed: every variable written in it, in the aliases,
   in the map of binding times and in the closures, those of symbols
   too. The binding times themselves are as they were: a variable keeps
   the time of its binding when it is renamed. Symbols are not
   renamed. *)
Definition rename_closure (r : renaming) (c : Known.closure) : Known.closure :=
  match c with
  | Known.Of_code code value_slots =>
    Known.Of_code code (map_values (rename_simple r) value_slots)
  | Known.Of_set slot function_decls value_slots =>
    Known.Of_set slot function_decls
      (map_values (rename_simple r) value_slots)
  end.

Definition rename_static_closure (r : renaming) (c : Known.static_closure)
    : Known.static_closure :=
  Known.make_static_closure (rename_closure r (Known.as_closure c))
    (Known.closure_symbols c).

Definition rename_known (r : renaming) (known : Known.t) : Known.t :=
  Known.make (rename_aliases r (Known.aliases known))
    (rename_binding_times r (Known.binding_times known))
    (Known.next_binding_time known)
    (map
       (fun entry =>
          (rename_variable r (fst entry), rename_closure r (snd entry)))
       (Known.closures known))
    (map_values (rename_static_closure r) (Known.static_closures known)).

(* The functions commute with renaming, with no premise: they compare
   names for equality only, and order them by when they were bound,
   which a renaming does not change. A variable that is bound again
   after the renaming was bound again before it. *)
Definition rename_bind_statement : Prop :=
  ∀ (r : renaming) (known : Known.t) (x : variable),
    Known.bind (rename_known r known) (rename_variable r x)
    = rename_known r (Known.bind known x).

Definition rename_under_let_statement : Prop :=
  ∀ (r : renaming) (known : Known.t) (x : variable) (d : named),
    Known.under_let (rename_known r known) (rename_variable r x)
      (rename_named r d)
    = rename_known r (Known.under_let known x d).

Definition rename_under_let_static_statement : Prop :=
  ∀ (r : renaming) (known : Known.t) (bindings : list static_binding),
    Known.under_let_static (rename_known r known)
      (map (rename_static_binding r) bindings)
    = rename_known r (Known.under_let_static known bindings).

Definition rename_follow_aliases_statement : Prop :=
  ∀ (r : renaming) (known : Known.t) (e : expr),
    follow_aliases (rename_known r known) (rename_expr r e)
    = rename_expr r (follow_aliases known e).

(* From nothing known, which is what the validator applies. *)
Definition rename_follow_aliases_empty_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    follow_aliases Known.empty (rename_expr r e)
    = rename_expr r (follow_aliases Known.empty e).
