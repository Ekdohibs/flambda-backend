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

(* In the body of code, what [my_closure] is known to be from its code
   is used: a call of it or of another closure of its set is made
   direct, and a value slot that every such set gives one symbol or
   constant is read.

   This file is to be reviewed: [sibling_code] and [slot_held], which
   say what is known; [know_code_closures] and what it is made of;
   [groups_whole]; [know_own_closures], which is to be extracted and
   run by the validation of middle_end/flambda2/tests/semantics/; and
   the statements.

   What Simplify does. It simplifies the code of a function where it
   simplifies the set of closures that the function is of, and gives
   [my_closure], in that code, the type of the closure
   (simplify/simplify_set_of_closures.ml, [dacc_inside_function]): the
   code of each function slot of the set, and the types of what the
   set puts in its value slots. So in that code the projection of
   another function slot of the set from [my_closure] is a closure of
   known code, a call of it is direct, and the projection of a value
   slot that the set gives a symbol or a constant is that symbol or
   constant. Simplify does it for one set, under the new code that it
   gives that set.

   The code of the model is bound once and may be the code of several
   sets. So this goes by the code and the function slot, and asks that
   every set of the unit which has the code at that function slot
   agrees (FlambdaSetsOfCodeSpec.sets_made_have, which is checked;
   nothing is guessed). In the body of code [id] that is no stub and
   whose [my_closure] is not also a parameter, [my_closure] is known
   to be a closure of the code [id]: a call finds the body that it
   runs by the code of its callee, and binds [my_closure] to the
   callee. Of a variable [x] known to be a closure of code [c]:

   - an indirect call of [x] is the direct call of [c];
   - a variable bound to [Project_function_slot f g (Var x)] is known
     to be a closure of the code that every set with [c] at [f] has
     at [g], if there is such a set and they all have the same
     ([sibling_code]);
   - [Project_value_slot_from f slot (Var x)] is the symbol or
     constant that every set with [c] at [f] gives [slot], if there
     is such a set, they all give the same and, for a symbol, it is
     in scope ([slot_held]).

   Nothing else is rewritten, and only in the bodies of code. What is
   known of a variable is its code, a code id; going under a binder
   of the variable, it is forgotten.

   The result refines the unit run on its own, from its imports, and
   followed by what may follow a unit, which is the observers of
   FlambdaObserversSpec.v at the value slots given, run from the
   imports (the three statements of [know_own_closures]); and not
   under all environments, nor followed by any other term: a closure
   from elsewhere may have the code [c] at [f] beside other code, or
   with other values in its value slots. What makes it right is that
   every closure of the run of a unit is of a set that the unit makes
   (FlambdaSetsOfCodeSpec.state_sets_have), and that the body of code
   runs only where [my_closure] is a closure of that code
   (FlambdaContextWhereSpec.let_static_code_refines_from_called_statement).
   A projection of a function slot from a symbol gives the symbol
   that the static set has for that slot, which is a closure of the
   set only where the group that binds the set binds all of it; so
   the rewrite is made only of a unit whose groups do
   ([groups_whole]), and is the identity of any other.

   The result has its code closed where the unit has, and no free
   variable or continuation that the unit has not
   ([know_own_closures_names_statement]); and the rewrite commutes
   with renaming ([know_own_closures_equivariant_statement]).

   The proofs are in FlambdaClosureOfCodeProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaBindersSpec FlambdaNominalSpec
  FlambdaWellFormedSpec FlambdaStaticsInScopeSpec FlambdaUnusedStaticsSpec
  FlambdaKnownSymbolsSpec FlambdaKnownSlotsSpec FlambdaSetsOfCodeSpec.
From OxCamlFlambda2 Require Import FlambdaImportsSpec FlambdaObserversSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known of the sets that have a code at a function slot            *)
(* ======================================================================== *)

(* The first of the sets that has [c] at [f]. *)
Definition set_with_code (sets : list set_made) (c : code_id)
    (f : function_slot) : option set_made :=
  find (fun set => has_code_at c f (fst set)) sets.

(* The code that every set with [c] at [f] has at [g], if there is such
   a set and they agree. *)
Definition sibling_code (sets : list set_made) (c : code_id)
    (f g : function_slot) : option code_id :=
  match set_with_code sets c f with
  | Some set =>
    match lookup g (fst set) with
    | Some d =>
      if sets_made_have c f (Code_at g d) sets then Some d else None
    | None => None
    end
  | None => None
  end.

(* The symbol or constant that every set with [c] at [f] gives the
   value slot, if there is such a set and they agree. *)
Definition slot_held (sets : list set_made) (c : code_id)
    (f : function_slot) (slot : value_slot) : option simple :=
  match set_with_code sets c f with
  | Some set =>
    match lookup slot (snd set) with
    | Some a =>
      if closed_simple a && sets_made_have c f (Slot_holds slot a) sets
      then Some a
      else None
    | None => None
    end
  | None => None
  end.

(* ======================================================================== *)
(* The rewrite of the body of code                                          *)
(* ======================================================================== *)

(* The variables known to be closures, each with the code of the
   closure. *)
Definition closure_codes : Type := finite_map code_id.

(* The code of the closure that a named expression gives, where it is
   known: the projection of a function slot from a variable that is
   known. *)
Definition closure_code_made (sets : list set_made) (known : closure_codes)
    (d : named) : option code_id :=
  match d with
  | Project_function_slot move_from move_to (Var x) =>
    match lookup x known with
    | Some c => sibling_code sets c move_from move_to
    | None => None
    end
  | Project_function_slot _ _ (Symbol _ | Const _) | Simple _ | Prim _ _
  | Set_of_closures _ _ | Closure_of_set _ _ _ | Project_value_slot _ _
  | Project_value_slot_from _ _ _ =>
    None
  end.

Definition codes_under_let (sets : list set_made) (known : closure_codes)
    (x : variable) (d : named) : closure_codes :=
  match closure_code_made sets known d with
  | Some c => (x, c) :: unbind x known
  | None => unbind x known
  end.

(* The projection of a value slot from a variable that is known is what
   every set with its code gives the slot, unless that is a symbol that
   is not in scope: [syms] are the symbols in scope. *)
Definition read_code_slot (sets : list set_made) (syms : list symbol)
    (known : closure_codes) (d : named) : named :=
  match d with
  | Project_value_slot_from project_from slot (Var x) =>
    match lookup x known with
    | Some c =>
      match slot_held sets c project_from slot with
      | Some a => if simple_statics_in_scope syms a then Simple a else d
      | None => d
      end
    | None => d
    end
  | Project_value_slot_from _ _ (Symbol _ | Const _) | Simple _ | Prim _ _
  | Set_of_closures _ _ | Closure_of_set _ _ _ | Project_value_slot _ _
  | Project_function_slot _ _ _ =>
    d
  end.

(* An indirect call of a variable that is known is a direct call of its
   code. *)
Definition call_code_closure (known : closure_codes) (ap : apply_expr)
    : apply_expr :=
  match ap_call_kind ap, ap_callee ap with
  | Indirect, Var x =>
    match lookup x known with
    | Some c =>
      Apply_expr (ap_callee ap) (Direct c) (ap_args ap) (ap_continuation ap)
        (ap_exn_continuation ap) (ap_exn_extra_args ap) (ap_args_arity ap)
        (ap_return_arity ap)
    | None => ap
    end
  | Indirect, (Symbol _ | Const _) | Direct _, _ => ap
  end.

(* What is known once the variables given are bound again: nothing
   of them. *)
Fixpoint forget_code_vars (xs : list variable) (known : closure_codes)
    : closure_codes :=
  match xs with
  | [] => known
  | x :: xs => forget_code_vars xs (unbind x known)
  end.

Fixpoint know_in_body (sets : list set_made) (syms : list symbol)
    (known : closure_codes) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind (read_code_slot sets syms known d)
      (know_in_body sets syms (codes_under_let sets known x d) body)
  | Let_static bindings body =>
    Let_static bindings (know_in_body sets syms known body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (know_in_body_handler sets syms known h)
      (know_in_body sets syms known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, know_in_body_handler sets syms known h))
         handlers)
      (know_in_body sets syms known body)
  | Apply ap => Apply (call_code_closure known ap)
  | Apply_cont _ | Switch _ _ | Invalid => e
  end
with know_in_body_handler (sets : list set_made) (syms : list symbol)
    (known : closure_codes) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (know_in_body sets syms (forget_code_vars params known) handler)
  end.

(* The code [id]: in its body, [my_closure] is a closure of [id],
   unless the code is that of a stub or [my_closure] is also one of
   its parameters. *)
Definition know_in_code (sets : list set_made) (syms : list symbol)
    (id : code_id) (c : code) : code :=
  match c with
  | Code params my_closure ret exn metadata body =>
    if cm_is_stub metadata || among my_closure params then c
    else
      Code params my_closure ret exn metadata
        (know_in_body sets syms [(my_closure, id)] body)
  end.

(* The code that a static binding binds. *)
Definition know_code_in_binding (sets : list set_made) (syms : list symbol)
    (binding : static_binding) : static_binding :=
  match binding with
  | Bind_code id c => Bind_code id (know_in_code sets syms id c)
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end.

(* Every code that the expression binds, in handlers too; not the code
   that the body of code binds. The symbols in scope are those of
   FlambdaStaticsInScopeSpec.expr_statics_in_scope, as in
   FlambdaKnownSlotsSpec.read_slots. *)
Fixpoint know_code_closures (sets : list set_made) (syms : list symbol)
    (e : expr) {struct e} : expr :=
  match e with
  | Let x kind d body => Let x kind d (know_code_closures sets syms body)
  | Let_static bindings body =>
    let all := flat_map symbols_of_binding bindings ++ syms in
    Let_static (map (know_code_in_binding sets all) bindings)
      (know_code_closures sets all body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k
      (know_code_in_handler sets
         (static_names symbols_of_binding body ++ syms) h)
      (know_code_closures sets syms body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k,
             know_code_in_handler sets
               (static_names symbols_of_binding body ++ syms) h))
         handlers)
      (know_code_closures sets syms body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with know_code_in_handler (sets : list set_made) (syms : list symbol)
    (h : continuation_handler) {struct h} : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (know_code_closures sets syms handler)
  end.

(* Every group of static bindings of the expression, in handlers and
   in code too, binds the static sets of closures that it binds whole
   (FlambdaWellFormedSpec.static_set_whole): each symbol of a set to
   the closure of the set at that symbol's function slot. *)
Fixpoint groups_whole (e : expr) : bool :=
  match e with
  | Let _ _ _ body => groups_whole body
  | Let_static bindings body =>
    forallb (static_set_whole bindings) bindings
    && forallb binding_groups_whole bindings
    && groups_whole body
  | Let_cont_non_recursive _ h body =>
    handler_groups_whole h && groups_whole body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_groups_whole h) handlers
    && groups_whole body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_groups_whole (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => groups_whole handler
  end
with binding_groups_whole (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_groups_whole c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_groups_whole (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => groups_whole body
  end.

(* The unit with what the closures of its code are known to be used,
   from the sets that it makes, where its groups bind their static
   sets whole; the unit as it is otherwise. No symbol is in scope to
   begin with: those that the unit imports are not known here. *)
Definition know_own_closures (e : expr) : expr :=
  if groups_whole e then know_code_closures (sets_made e) [] e else e.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* A unit whose groups bind their static sets whole, with what the
   closures of its code are known to be from sets that include those
   that it makes, refines the unit, run on its own; whatever the
   symbols taken to be in scope. *)
Definition know_code_closures_program_refines_statement : Prop :=
  ∀ (sets : list set_made) (syms : list symbol) (e : expr),
    incl (sets_made e) sets ->
    groups_whole e = true ->
    program_refines e (know_code_closures sets syms e).

(* So what [know_own_closures] gives refines the unit, run on its
   own. *)
Definition know_own_closures_program_refines_statement : Prop :=
  ∀ e : expr, program_refines e (know_own_closures e).

(* The same of a unit that imports symbols, each run on its own from
   the same imports (FlambdaImportsSpec.unit_refines): the static
   environment of the imports has no code, and an import is no
   closure. *)
Definition know_own_closures_refines_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (e : expr),
    unit_refines imports e (know_own_closures e).

(* The same of the unit followed by what may follow a unit, and by
   nothing else: the unit and the unit rewritten, run from one
   continuation environment of the observers of
   FlambdaObserversSpec.v at the value slots given
   (FlambdaObserversSpec.observers) and from the static environment
   of the imports. What follows is the same term in the
   two runs: it has no code. It makes no closure, since it names no
   code id, so every closure of the run is still of a set that the
   unit makes. *)
Definition know_own_closures_refines_observed_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (slots : list value_slot)
    (conts : cont_env) (e : expr),
    observers slots conts ->
    refines (State e [] conts (imports_statics imports) empty_heap)
      (State (know_own_closures e) [] conts (imports_statics imports)
         empty_heap).

(* The names of the result. Where the code of the unit is closed
   (FlambdaBindersSpec.code_closed), so is that of the result, and the
   variables and the continuations that are free in the result are
   free in the unit: a call keeps its callee, its arguments and its
   continuations, and what is put in the place of a projection is a
   symbol or a constant. *)
Definition know_own_closures_names_statement : Prop :=
  ∀ e : expr,
    code_closed e = true ->
    code_closed (know_own_closures e) = true
    ∧ incl (free_variables_of_expr (know_own_closures e))
        (free_variables_of_expr e)
    ∧ incl (free_continuations_of_expr (know_own_closures e))
        (free_continuations_of_expr e).

(* The rewrite commutes with renaming, which renames no symbol, no code
   id and no slot: the sets that the unit renamed makes are those of
   the unit with the variables in their value slots renamed, the tests
   answer the same of them, and what is put in the place of a
   projection has no variable in it. *)
Definition know_own_closures_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    know_own_closures (rename_expr r e) = rename_expr r (know_own_closures e).
