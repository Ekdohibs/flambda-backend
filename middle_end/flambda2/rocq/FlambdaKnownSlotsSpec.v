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

(* A value slot that every set of closures of the unit gives the same
   symbol or constant is that symbol or constant wherever it is read.

   This file is to be reviewed: [slots_made] and [slots_known], which
   say what is known; [slots_as], the check; [read_slots] and
   [read_slot], the rewrite, against the rule of scope of
   FlambdaStaticsInScopeSpec.v; [read_known_slots], which is extracted
   and run by the validation of middle_end/flambda2/tests/semantics/;
   [read_known_slots_anywhere], the same with every symbol that the
   unit binds taken to be in scope, which the validation runs after
   its last lifting; and the statements, the last six of which are
   the rewrites computed on small units: a slot read, a slot that one
   set gives a variable, a symbol that is not in scope, a projection
   in the code of a stub, a set that the code of a stub makes, and
   the symbol that is not in scope read all the same.

   What Simplify does. It simplifies the code of a function where it
   simplifies the set of closures that the function is of, and gives
   [my_closure], in that code, the type of an alias of the name that
   the closure is bound to there (simplify/simplify_set_of_closures.ml,
   [dacc_inside_function]). That name has the type of the closure,
   whose value slots have the types of what the set puts in them
   (simplify/simplify_set_of_closures_context.ml,
   [compute_closure_types_inside_functions]). So
   where a function that stays a function makes a closure that uses a
   function of the module, or itself, the value slot of the closure
   holds a symbol once those are lifted, and in the code of the closure
   the projection of that slot from [my_closure] becomes the symbol
   (simplify/simplify_unary_primitive.ml, [simplify_project_value_slot]):
   the code calls the symbol.

   That is right of the code as Simplify rewrites it for that one set.
   The code of the model is bound once and may be the code of several
   closures, and nothing in the model says which set a closure that is
   called was made by (FlambdaClosuresSpec.v, "The set that a piece of
   code belongs to"). So what is here does not go by the code. It goes
   by the slot, and asks more than Simplify needs:

   - every set of closures that the unit makes, by a [Let] or
     statically, in code and in handlers too, and that has the value
     slot, gives it the same symbol or constant. That is checked
     ([slots_as]), for what the unit itself is read to make
     ([slots_known] of [slots_made]): nothing is guessed, and nothing
     is believed of what is read. A slot that some set gives a
     variable is never read, whatever the other sets give it: the
     value of a variable is computed when the unit runs.
   - the unit is run on its own, from the empty environments, where
     nothing that it imports is bound. Every closure of its run is
     then made by one of those sets, or is a symbol that the unit
     binds to one, so that a projection of the slot that has
     a value has that symbol or constant for its value, whatever
     closure it is from ([slots_as_kept_statement],
     [read_slot_sound_statement]). Under other environments, or with a
     closure of another unit, this fails: a closure from elsewhere may
     hold another value in a slot of the same name. So this is no
     rewrite under all environments, and the statements are of the
     run from the empty environments
     (FlambdaUnusedStaticsSpec.program_refines).
   - the symbol is in scope where it is put
     (FlambdaStaticsInScopeSpec.statics_in_scope, row 4 of
     requests/C0-well-formed.md). The rewrite goes down the unit with
     the symbols in scope, as that check does, and where the symbol
     that the slot holds is not among them it leaves the projection as
     it is. A unit that passes row 4 passes it once rewritten
     ([read_slots_in_scope_statement]). The symbols that the unit
     imports are not known here and are taken to be out of scope: a
     slot that holds one is not read.

   So the rewrite is the scoped one, and it reads less than Simplify
   does. The unit that shows the difference is
   capture_fun_bound_in_handler_run.ml, in
   middle_end/flambda2/tests/semantics/ocaml/: the closure of a
   function holds a value that a test chooses when the module runs, so
   its symbol is bound in the handler of that test, and the code of a
   closure that calls the function is bound at the head of the unit,
   where that symbol is not in scope. The projection stays there;
   Simplify, which simplifies that code where the closure is made,
   puts the symbol. [read_known_slots_anywhere], below, reads it, and
   with it the unit is validated.

   [read_slots known syms e] replaces, in the whole of [e], in code and
   in handlers too, the defining expression of a [Let] that is the
   projection of a value slot that [known] lists, from any closure, by
   the symbol or constant listed, where that is in scope. The [Let]
   stays: its variable is then an alias, which other passes follow
   (FlambdaAliasesSpec.follow_aliases) and remove
   (FlambdaNormaliseSpec.remove_unused_lets). The set of closures keeps
   the slot.

   The code of a stub is left as it is (FlambdaModel.code_is_stub, the
   compiler's [Code.stub]). Unless [-flambda2-simplify-stubs] is
   given, and by default it is not (driver/oxcaml_flags.ml), Simplify
   does not simplify the code of a stub where it simplifies the set of
   closures that the code is of (simplify/simplify_set_of_closures.ml,
   [simplify_function]). It simplifies that code where it is bound
   (simplify/simplify_static_const.ml, by
   [Simplify_set_of_closures.simplify_static_stub_function]), and
   there [my_closure] has an unknown type ([dacc_inside_function], the
   case of no function slot): a projection from it stays a projection.
   The sets of closures that the code of a stub makes are read with
   all the others, for what is known and for the check: a closure that
   a stub makes may reach code that is no stub, which reads its slots.
   Nothing is believed of the flag: leaving a projection where it is
   refines whatever code the flag is set on (discrepancy C3-11).

   The unit rewritten refines the unit, run on its own
   ([read_known_slots_program_refines_statement]). It is not an
   equivalence: a projection that has no value in the unit (from what
   is not a closure, or from a closure that has not got the slot) has
   one once rewritten.

   The same holds from a static environment that binds something
   already, where the code in it makes its sets of closures as what
   is known says and what it binds its symbols to has its value
   slots so ([read_slots_refines_from_statement]). A unit is run so
   from its imports (FlambdaImportsSpec.imports_statics), which are
   no code and no closure:
   [read_known_slots_refines_imports_statement].

   It holds too of the unit followed by something
   ([read_known_slots_refines_observed_statement]): what may follow a
   unit (FlambdaObserversSpec.observers) binds no code and names no
   code id, so it makes no closure, and the closures that it has are
   those that the unit gave it.

   The test of scope is there for row 4 and for nothing else: the
   refinement holds whatever symbols are taken to be in scope. Simplify
   reads the slot also where the code is bound above the symbol in the
   text of the unit, as it binds the new version of the code where the
   set of closures is. [read_known_slots_anywhere] does the same: it
   takes every symbol that the unit binds to be in scope. What it
   gives need not pass row 4, so it is for where nothing that follows
   relies on row 4 of the term without testing it; what tests it may
   then refuse. It refines the unit as [read_known_slots] does
   ([read_known_slots_anywhere_program_refines_statement], and the
   two statements after it).

   What is not here: a slot that holds a variable; a slot that two sets
   give different symbols, where the code that reads it is the code of
   one of them only (Simplify knows that; this file would need the test
   FlambdaClosuresSpec.sets_are for it); the removal of a slot that
   nothing reads any more from a set that is not static
   (FlambdaUnusedSlotsSpec.v removes them from the static ones).

   The proofs are in FlambdaKnownSlotsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaWellFormedSpec FlambdaStaticsInScopeSpec FlambdaUnusedStaticsSpec
  FlambdaKnownSymbolsSpec.
From OxCamlFlambda2 Require Import FlambdaImportsSpec FlambdaObserversSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known                                                            *)
(* ======================================================================== *)

(* Value slots, each with the symbol or the constant that it holds. A
   slot may be listed more than once; the first is used. *)
Definition known_slots : Type := list (value_slot * simple).

(* Whether two simples are the same. *)
Definition same_simple (a a' : simple) : bool :=
  if simple_eq_dec a a' then true else false.

(* Every slot listed holds a symbol or a constant. *)
Definition closed_slots (known : known_slots) : bool :=
  forallb (fun entry => closed_simple (snd entry)) known.

(* The value slots that a defining expression, or a static constant,
   gives the set of closures that it makes, with what it puts in
   them. *)
Definition named_slots_made (d : named) : list (value_slot * simple) :=
  match d with
  | Set_of_closures _ value_slots | Closure_of_set _ _ value_slots =>
    value_slots
  | Simple _ | Prim _ _ | Project_value_slot _ _
  | Project_value_slot_from _ _ _ | Project_function_slot _ _ _ =>
    []
  end.

Definition static_const_slots_made (const : static_const)
    : list (value_slot * simple) :=
  match const with
  | Static_const_set_of_closures _ value_slots
  | Static_const_closure_of_set _ _ _ value_slots =>
    value_slots
  | Static_const_block _ _ | Static_const_boxed_number _
  | Static_const_immutable_string _
  | Static_const_immutable_value_array _ =>
    []
  end.

(* The same of all the sets that an expression makes, in handlers and
   in code too. *)
Fixpoint slots_made (e : expr) : list (value_slot * simple) :=
  match e with
  | Let _ _ d body => named_slots_made d ++ slots_made body
  | Let_static bindings body =>
    flat_map binding_slots_made bindings ++ slots_made body
  | Let_cont_non_recursive _ h body =>
    handler_slots_made h ++ slots_made body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => handler_slots_made h) handlers
    ++ slots_made body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_slots_made (h : continuation_handler)
    : list (value_slot * simple) :=
  match h with
  | Handler _ _ handler => slots_made handler
  end
with binding_slots_made (binding : static_binding)
    : list (value_slot * simple) :=
  match binding with
  | Bind_code _ c => code_slots_made c
  | Bind_deleted_code _ => []
  | Bind_symbol _ const => static_const_slots_made const
  end
with code_slots_made (c : code) : list (value_slot * simple) :=
  match c with
  | Code _ _ _ _ _ body => slots_made body
  end.

(* Of the slots made, those that hold a symbol or a constant, the same
   wherever the slot is made. *)
Definition slots_known (made : list (value_slot * simple)) : known_slots :=
  filter
    (fun '(slot, a) =>
       closed_simple a
       && forallb
            (fun '(slot', a') => negb (Pos.eqb slot' slot) || same_simple a' a)
            made)
    made.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

(* The value slots of a set are as [known] says: each of them that it
   lists holds what it lists for it. *)
Definition value_slots_as (known : known_slots)
    (value_slots : list (value_slot * simple)) : bool :=
  forallb
    (fun '(slot, a) =>
       match lookup slot known with
       | Some a' => same_simple a a'
       | None => true
       end)
    value_slots.

(* Every set of closures that the expression makes, in handlers and in
   code too, has its value slots as [known] says. *)
Fixpoint slots_as (known : known_slots) (e : expr) : bool :=
  match e with
  | Let _ _ d body =>
    value_slots_as known (named_slots_made d) && slots_as known body
  | Let_static bindings body =>
    forallb (binding_slots_as known) bindings && slots_as known body
  | Let_cont_non_recursive _ h body =>
    handler_slots_as known h && slots_as known body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_slots_as known h) handlers
    && slots_as known body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_slots_as (known : known_slots) (h : continuation_handler)
    : bool :=
  match h with
  | Handler _ _ handler => slots_as known handler
  end
with binding_slots_as (known : known_slots) (binding : static_binding)
    : bool :=
  match binding with
  | Bind_code _ c => code_slots_as known c
  | Bind_deleted_code _ => true
  | Bind_symbol _ const =>
    value_slots_as known (static_const_slots_made const)
  end
with code_slots_as (known : known_slots) (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => slots_as known body
  end.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* The projection of a value slot that [known] lists is what it lists,
   unless that is a symbol that is not in scope: [syms] are the symbols
   in scope. Nothing here asks that what is listed be a symbol or a
   constant, and a variable listed would pass the test of scope:
   [read_known_slots] and the statements of refinement ask it of
   [known] ([closed_slots]). *)
Definition read_slot (known : known_slots) (syms : list symbol) (d : named)
    : named :=
  match d with
  | Project_value_slot slot _ | Project_value_slot_from _ slot _ =>
    match lookup slot known with
    | Some a => if simple_statics_in_scope syms a then Simple a else d
    | None => d
    end
  | Simple _ | Prim _ _ | Set_of_closures _ _ | Closure_of_set _ _ _
  | Project_function_slot _ _ _ =>
    d
  end.

(* Every [Let] of the expression, in handlers and in code too, but for
   the code of a stub, which is left as it is. The symbols in scope
   are those of FlambdaStaticsInScopeSpec.expr_statics_in_scope: in
   the code and in the body of a group, all that the group binds; in a
   handler of a [Let_cont], what the body of the [Let_cont] binds. *)
Fixpoint read_slots (known : known_slots) (syms : list symbol) (e : expr)
    {struct e} : expr :=
  match e with
  | Let x kind d body =>
    Let x kind (read_slot known syms d) (read_slots known syms body)
  | Let_static bindings body =>
    let all := flat_map symbols_of_binding bindings ++ syms in
    Let_static (map (read_slots_in_binding known all) bindings)
      (read_slots known all body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k
      (read_slots_in_handler known
         (static_names symbols_of_binding body ++ syms) h)
      (read_slots known syms body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k,
             read_slots_in_handler known
               (static_names symbols_of_binding body ++ syms) h))
         handlers)
      (read_slots known syms body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with read_slots_in_handler (known : known_slots) (syms : list symbol)
    (h : continuation_handler) {struct h} : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (read_slots known syms handler)
  end
with read_slots_in_binding (known : known_slots) (syms : list symbol)
    (binding : static_binding) {struct binding} : static_binding :=
  match binding with
  | Bind_code id c =>
    if code_is_stub c then binding
    else Bind_code id (read_slots_in_code known syms c)
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end
with read_slots_in_code (known : known_slots) (syms : list symbol) (c : code)
    {struct c} : code :=
  match c with
  | Code params my_closure ret exn metadata body =>
    Code params my_closure ret exn metadata (read_slots known syms body)
  end.

(* The unit with the value slots that it is read to give one symbol or
   constant read wherever they are projected, if the check passes of
   what was read; the unit as it is otherwise. No symbol is in scope to
   begin with: those that the unit imports are not known here. *)
Definition read_known_slots (e : expr) : expr :=
  let known := slots_known (slots_made e) in
  if closed_slots known && slots_as known e
  then read_slots known [] e
  else e.

(* The same with every symbol that the unit binds taken to be in scope
   (FlambdaWellFormedSpec.static_names): a projection is read also in
   code that is bound above the symbol that the slot holds. What this
   gives need not pass row 4. *)
Definition read_known_slots_anywhere (e : expr) : expr :=
  let known := slots_known (slots_made e) in
  if closed_slots known && slots_as known e
  then read_slots known (static_names symbols_of_binding e) e
  else e.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The value is that of the symbol or constant. *)
Definition value_is (a : simple) (v : value) : bool :=
  match v with
  | Value_symbol s => same_simple (Symbol s) a
  | Value_const c => same_simple (Const c) a
  | Value_closure _ _ | Value_closure_of_set _ _ _ | Value_block _ _
  | Value_immutable_array _ | Value_boxed_number _ | Value_location _
  | Value_partial_application _ _ _ =>
    false
  end.

(* The values of the value slots of a closure are as [known] says:
   each slot that it lists has the value of what it lists for it. *)
Definition slot_values_as (known : known_slots)
    (slot_values : list (value_slot * value)) : bool :=
  forallb
    (fun '(slot, v) =>
       match lookup slot known with
       | Some a => value_is a v
       | None => true
       end)
    slot_values.

(* Every closure in the value has its value slots so, through the
   values of value slots, the fields of blocks and partial
   applications. A symbol is so whatever it is bound to: what it is
   bound to is asked to be so where it is bound
   ([allocated_slots_known]). *)
Fixpoint value_slots_known (known : known_slots) (v : value) : bool :=
  match v with
  | Value_const _ | Value_symbol _ | Value_boxed_number _ | Value_location _ =>
    true
  | Value_closure _ slot_values | Value_closure_of_set _ _ slot_values =>
    slot_values_as known slot_values
    && forallb (fun '(_, contents) => value_slots_known known contents)
         slot_values
  | Value_block _ fields | Value_immutable_array fields =>
    forallb (value_slots_known known) fields
  | Value_partial_application callee args _ =>
    value_slots_known known callee
    && forallb (value_slots_known known) args
  end.

Definition allocated_slots_known (known : known_slots)
    (allocated : statically_allocated_value) : bool :=
  match allocated with
  | Static_closure _ slot_values | Static_closure_of_set _ _ _ slot_values =>
    slot_values_as known slot_values
    && forallb (fun '(_, contents) => value_slots_known known contents)
         slot_values
  | Static_block _ fields => forallb (value_slots_known known) fields
  | Static_immutable_array elements =>
    forallb (value_slots_known known) elements
  | Static_boxed_number contents => value_slots_known known contents
  | Static_string _ | Static_imported _ => true
  end.

(* Every value that the state holds is so: those of its variables, of
   the environments that its continuations keep and of the cells of its
   heap, and what each symbol is bound to
   (FlambdaSemanticsSpec.state_values_all). *)
Definition state_slots_known (known : known_slots) (s : state) : Prop :=
  state_values_all (fun _ v => value_slots_known known v = true)
    (fun _ allocated => allocated_slots_known known allocated = true) s.

(* Where every set of closures that a unit makes has its value slots as
   [known] says, and [known] lists symbols and constants only, every
   state of the run of the unit on its own holds closures with those
   slots so. Nothing else is asked of the unit; it need not be
   well-formed. *)
Definition slots_as_kept_statement : Prop :=
  ∀ (known : known_slots) (e : expr) (n : nat) (s : state),
    closed_slots known = true ->
    slots_as known e = true ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    state_slots_known known s.

(* In such a state, a defining expression that has a value has the same
   value once rewritten, whatever the symbols in scope. *)
Definition read_slot_sound_statement : Prop :=
  ∀ (known : known_slots) (syms : list symbol) (s : state) (d : named)
    (v : value),
    state_slots_known known s ->
    eval_named (state_statics s) (state_vars s) d = Some v ->
    eval_named (state_statics s) (state_vars s) (read_slot known syms d)
    = Some v.

(* The rewrite makes no set of closures and changes none: a term passes
   the check before it if and only if it does after. *)
Definition read_slots_slots_as_statement : Prop :=
  ∀ (known known' : known_slots) (syms : list symbol) (e : expr),
    slots_as known' (read_slots known syms e) = slots_as known' e.

(* A unit whose sets of closures have their value slots as [known]
   says, with the projections of those slots read, refines the unit,
   run on its own; whatever the symbols taken to be in scope. *)
Definition read_slots_program_refines_statement : Prop :=
  ∀ (known : known_slots) (syms : list symbol) (e : expr),
    closed_slots known = true ->
    slots_as known e = true ->
    program_refines e (read_slots known syms e).

(* So what [read_known_slots] gives refines the unit, run on its
   own. *)
Definition read_known_slots_program_refines_statement : Prop :=
  ∀ e : expr, program_refines e (read_known_slots e).

(* The same from a static environment that binds something already:
   the code that it has makes its sets of closures as [known] says,
   and what it binds its symbols to has its value slots so. The heap
   is empty. *)
Definition read_slots_refines_from_statement : Prop :=
  ∀ (known : known_slots) (syms : list symbol) (e : expr)
    (statics : static_env),
    closed_slots known = true ->
    slots_as known e = true ->
    Forall (fun binding => code_slots_as known (snd binding) = true)
      (static_code statics) ->
    Forall
      (fun binding => allocated_slots_known known (snd binding) = true)
      (static_values statics) ->
    refines (State e [] [] statics empty_heap no_reads)
      (State (read_slots known syms e) [] [] statics empty_heap no_reads).

(* So what [read_known_slots] gives refines the unit, each run on its
   own from the same imports (FlambdaImportsSpec.unit_refines): the
   static environment of the imports has no code, and binds each
   symbol to something imported, which has no value slot. *)
Definition read_known_slots_refines_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (e : expr),
    unit_refines imports e (read_known_slots e).

(* The same of the unit followed by something: the unit and the unit
   rewritten, run from one continuation environment of what may follow
   a unit (FlambdaObserversSpec.observers: a way out, or a handler with
   no environment of its own that binds nothing static and names no
   symbol and no code id). What follows is the same term in the two
   runs: it is not rewritten. It may project a value slot from a
   closure that the unit gave it, and call one; it makes no closure,
   since it names no code id, and a set of closures that has no
   function has no value (FlambdaSemanticsSpec.eval_named). So every
   closure of the run is still made by a set of the unit or of the
   code that the static environment has. With no continuation this is
   [read_slots_refines_from_statement]. *)
Definition read_slots_refines_observed_statement : Prop :=
  ∀ (known : known_slots) (syms : list symbol) (e : expr) (conts : cont_env)
    (statics : static_env),
    closed_slots known = true ->
    slots_as known e = true ->
    Forall (fun binding => code_slots_as known (snd binding) = true)
      (static_code statics) ->
    Forall
      (fun binding => allocated_slots_known known (snd binding) = true)
      (static_values statics) ->
    observers [] conts ->
    refines (State e [] conts statics empty_heap no_reads)
      (State (read_slots known syms e) [] conts statics empty_heap no_reads).

(* So what [read_known_slots] gives refines the unit, the two followed
   by the same thing and run from the same imports, with the same
   exits. *)
Definition read_known_slots_refines_observed_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (conts : cont_env) (e : expr),
    observers [] conts ->
    refines (State e [] conts (imports_statics imports) empty_heap no_reads)
      (State (read_known_slots e) [] conts (imports_statics imports)
         empty_heap no_reads).

(* The same three of [read_known_slots_anywhere]: the refinement does
   not go by the symbols taken to be in scope. *)
Definition read_known_slots_anywhere_program_refines_statement : Prop :=
  ∀ e : expr, program_refines e (read_known_slots_anywhere e).

Definition read_known_slots_anywhere_refines_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (e : expr),
    unit_refines imports e (read_known_slots_anywhere e).

Definition read_known_slots_anywhere_refines_observed_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (conts : cont_env) (e : expr),
    observers [] conts ->
    refines (State e [] conts (imports_statics imports) empty_heap no_reads)
      (State (read_known_slots_anywhere e) [] conts
         (imports_statics imports) empty_heap no_reads).

(* The rewrite keeps row 4: where every symbol and code id that a term
   mentions is in scope, given the symbols [syms'] and the code ids
   [ids] in scope around it, the same holds of the term rewritten with
   some of [syms'] taken to be in scope. *)
Definition read_slots_in_scope_statement : Prop :=
  ∀ (known : known_slots) (syms syms' : list symbol) (ids : list code_id)
    (e : expr),
    incl syms syms' ->
    expr_statics_in_scope syms' ids e = true ->
    expr_statics_in_scope syms' ids (read_slots known syms e) = true.

Definition read_known_slots_in_scope_statement : Prop :=
  ∀ (imports : list symbol) (e : expr),
    statics_in_scope imports e = true ->
    statics_in_scope imports (read_known_slots e) = true.

(* The rewrite commutes with renaming, which renames no symbol, no
   constant and no value slot: the slots known are the same, and what
   is put in place of a projection has no variable in it. *)
Definition read_known_slots_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    read_known_slots (rename_expr r e) = rename_expr r (read_known_slots e).

(* So does [read_known_slots_anywhere]: a renaming renames no symbol,
   so the symbols that the unit binds are the same. *)
Definition read_known_slots_anywhere_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    read_known_slots_anywhere (rename_expr r e)
    = rename_expr r (read_known_slots_anywhere e).

(* ======================================================================== *)
(* Six units                                                                *)
(* ======================================================================== *)

(* What the rewrites make of six small units, computed. In each, the
   code [reader] is that of a closure with two value slots, which it
   projects from [my_closure]; the code [maker] makes that closure,
   with what it is given in the first slot and its parameter in the
   second. *)
Section Examples.

Let x : variable := 1%positive.
Let my_closure : variable := 2%positive.
Let first : variable := 3%positive.
Let second : variable := 4%positive.
Let made : variable := 5%positive.
Let ret : continuation := 1%positive.
Let exn : continuation := 2%positive.
Let done : continuation := 3%positive.
Let reader : code_id := 1%positive.
Let maker : code_id := 2%positive.
Let other_maker : code_id := 3%positive.
Let held : symbol := 1%positive.
Let first_slot : value_slot := 1%positive.
Let second_slot : value_slot := 2%positive.

Definition example_arity : arity := [Singleton Value].

(* The code of the closure, with the defining expression given for its
   first slot. *)
Definition example_reader (first_read : named) : code :=
  Code [x] my_closure ret exn
    (make_code_metadata example_arity example_arity)
    (Let first Value first_read
       (Let second Value (Project_value_slot second_slot (Var my_closure))
          (Apply_cont (Apply_cont_expr ret [Var first] None)))).

(* The code that makes the closure, with [in_first] in its first
   slot. *)
Definition example_maker (in_first : simple) : code :=
  Code [x] my_closure ret exn
    (make_code_metadata example_arity example_arity)
    (Let made Value
       (Set_of_closures reader [(first_slot, in_first); (second_slot, Var x)])
       (Apply_cont (Apply_cont_expr ret [Var made] None))).

Definition example_block : static_const := Static_const_block 0 [].

(* One group binds the two pieces of code and the symbol [held], which
   the closure holds in its first slot. *)
Definition example_read (first_read : named) : expr :=
  Let_static
    [Bind_code reader (example_reader first_read);
     Bind_code maker (example_maker (Symbol held));
     Bind_symbol held example_block]
    (Apply_cont (Apply_cont_expr done [Symbol held] None)).

(* The same with a second piece of code that makes the closure with its
   parameter in the first slot. *)
Definition example_variable : expr :=
  Let_static
    [Bind_code reader
       (example_reader (Project_value_slot first_slot (Var my_closure)));
     Bind_code maker (example_maker (Symbol held));
     Bind_code other_maker (example_maker (Var x));
     Bind_symbol held example_block]
    (Apply_cont (Apply_cont_expr done [Symbol held] None)).

(* The code of the closure bound in a group of its own, before the
   group that binds the symbol and the code that makes the closure. *)
Definition example_out_of_scope : expr :=
  Let_static
    [Bind_code reader
       (example_reader (Project_value_slot first_slot (Var my_closure)))]
    (Let_static
       [Bind_symbol held example_block;
        Bind_code maker (example_maker (Symbol held))]
       (Apply_cont (Apply_cont_expr done [Symbol held] None))).

(* Where every set gives the first slot the symbol, and the symbol is
   in scope in the code, the projection of the first slot is the
   symbol. The second slot holds a variable: its projection is left. *)
Definition read_known_slots_reads_statement : Prop :=
  read_known_slots
    (example_read (Project_value_slot first_slot (Var my_closure)))
  = example_read (Simple (Symbol held)).

(* Where one set gives the first slot a variable, nothing is read. *)
Definition read_known_slots_variable_statement : Prop :=
  read_known_slots example_variable = example_variable.

(* Where the symbol is not in scope in the code that projects the
   slot, the projection is left. The unit passes row 4. *)
Definition read_known_slots_out_of_scope_statement : Prop :=
  statics_in_scope [] example_out_of_scope = true
  ∧ read_known_slots example_out_of_scope = example_out_of_scope.

(* The same unit with the projection of the first slot read. *)
Definition example_out_of_scope_read : expr :=
  Let_static [Bind_code reader (example_reader (Simple (Symbol held)))]
    (Let_static
       [Bind_symbol held example_block;
        Bind_code maker (example_maker (Symbol held))]
       (Apply_cont (Apply_cont_expr done [Symbol held] None))).

(* With every symbol that the unit binds taken to be in scope, the
   projection is read there too, and what results does not pass
   row 4. *)
Definition read_known_slots_anywhere_out_of_scope_statement : Prop :=
  read_known_slots_anywhere example_out_of_scope = example_out_of_scope_read
  ∧ statics_in_scope [] example_out_of_scope_read = false.

(* The same code, marked as a stub. *)
Definition example_stub (c : code) : code :=
  match c with
  | Code params my_closure ret exn metadata body =>
    Code params my_closure ret exn (stub_code_metadata metadata) body
  end.

(* The first unit, with the code of the closure a stub. *)
Definition example_stub_reads : expr :=
  Let_static
    [Bind_code reader
       (example_stub
          (example_reader (Project_value_slot first_slot (Var my_closure))));
     Bind_code maker (example_maker (Symbol held));
     Bind_symbol held example_block]
    (Apply_cont (Apply_cont_expr done [Symbol held] None)).

(* Where the code that projects the slot is a stub, the projection is
   left, though every set gives the slot the symbol and the symbol is
   in scope. *)
Definition read_known_slots_in_stub_statement : Prop :=
  read_known_slots example_stub_reads = example_stub_reads.

(* The second unit, with the code that gives the slot a variable a
   stub. *)
Definition example_stub_makes : expr :=
  Let_static
    [Bind_code reader
       (example_reader (Project_value_slot first_slot (Var my_closure)));
     Bind_code maker (example_maker (Symbol held));
     Bind_code other_maker (example_stub (example_maker (Var x)));
     Bind_symbol held example_block]
    (Apply_cont (Apply_cont_expr done [Symbol held] None)).

(* A set that the code of a stub makes counts as any other: where it
   gives the slot a variable, nothing is read, in code that is no stub
   either. *)
Definition read_known_slots_stub_makes_statement : Prop :=
  read_known_slots example_stub_makes = example_stub_makes.

End Examples.
