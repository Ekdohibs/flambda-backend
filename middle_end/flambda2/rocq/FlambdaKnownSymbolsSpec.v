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

(* What the unit binds a symbol to is known wherever the symbol is used:
   a primitive or a projection applied to symbols bound to constants is
   replaced by its value.

   This file is to be reviewed: [known_bindings], [fold_known] and what
   they are made of, against what Simplify does (below); [bound_as], the
   check; [fold_known_unit], which is extracted and run by the validation
   of middle_end/flambda2/tests/semantics/; and the statements.

   What Simplify does. A symbol has a type from where it is bound to the
   end of the unit, in code too, and that type says what it is bound to:
   the closures of a static set with their symbols and what their value
   slots hold, the tag and the fields of a static block, the contents of
   a boxed number (simplify/simplify_static_const.ml and
   simplify/lifting/, which give the symbols of a group their types
   before the rest of the unit is simplified). So, wherever such a
   symbol is an argument: the projection of a function slot from the
   symbol of a closure is the symbol of the other closure of the set
   (simplify/simplify_unary_primitive.ml,
   [simplify_project_function_slot]); the projection of a value slot is
   what the slot holds ([simplify_project_value_slot]); a load from a
   static block is its field, and [Is_int] and [Get_tag] of one are
   their answers ([simplify_immutable_block_load] and the like). In the
   code of a function whose closure is a symbol, [my_closure] is that
   symbol (FlambdaMyClosureSpec.v), so the code of two functions that
   call each other comes to call through the two symbols. And an
   indirect call of such a symbol is made a direct call of the code of
   its closure (simplify/simplify_apply_expr.ml,
   [simplify_function_call], which reads the closure off the type of
   the callee): so where a function that is given a function of the
   unit is inlined, the call of its argument is direct, and may be
   inlined in turn.

   What is here. [known_bindings e] are the symbols that the unit [e]
   binds, outside code, to a constant with no variable in it: one whose
   fields or value slots are symbols and constants. Such a constant has
   one value whatever the environment ([known_statics]). [fold_known]
   replaces, in the whole unit, in code and in handlers too, the
   defining expression of a [Let] whose arguments are constants and
   such symbols, and which has under [known_statics] a value that is a
   constant or a symbol, by that value ([fold_known_named]). Of a
   closure of a static set whose value slots hold variables, less is
   known, and it is known whatever they hold: its function slot, and
   the symbols and the functions of the set ([known_sets]). That is
   what the projection of a function slot reads, so one from such a
   symbol is replaced by the symbol of the other closure
   ([fold_function_slot]); and it says which code the closure is of, so
   an indirect call of such a symbol is made a direct call of that code
   ([fold_known_apply]). Of a static block one of whose fields is a
   variable, the tag and the other fields are known, whatever the
   variable holds ([known_sets] lists the block with [unknown_field]
   for such a field): a load of a field that is a symbol or a constant
   is replaced by it ([fold_block_load]), as Simplify replaces one from
   the type of the block
   (simplify/simplify_unary_primitive.ml,
   [simplify_immutable_block_load]). That is how the code of a function
   comes to call a function of a module of the unit by its symbol,
   where the module also holds a value computed when the unit runs.
   The value
   is computed by the evaluation of the machine itself
   (FlambdaSemanticsSpec.eval_named), so there is no list of primitives
   here, and nothing to keep in step with them. The one primitive that
   the normalisation never folds is not folded here either
   (FlambdaNormaliseSpec.prim_may_fold: [Opaque_identity], which
   Simplify leaves alone). Nor is the load of bytes from a string
   ([reads_string]): Simplify does not compute it
   ([Simplify_binary_primitive.simplify_string_or_bigstring_load]), so
   the term after keeps the load, and with the load kept here the two
   terms have it. No primitive of the model has an effect;
   one that allocates something with an identity, or reads what may
   change, must be left out too when the model has them.

   It is right where each symbol, if it is bound when the [Let] runs,
   is bound to that constant. Under other environments it may be bound
   to something else, so this is no rewrite under all environments. For
   a unit run on its own it is enough that every static binding of such
   a symbol in the unit binds it to the one constant (for a symbol of
   [known_sets]: to a closure at that slot of that set, or to a block
   that is the same block once each field that is a variable is
   replaced by [unknown_field]), which is what
   [bound_as] checks; the symbol is then bound to it or to nothing, and
   where it is bound to nothing the defining expression has no value in
   the unit as it was (every operation on a symbol bound to nothing is
   undefined, decision D8). So the unit folded refines the unit, run on
   its own ([fold_known_unit_program_refines_statement]); it is not an
   equivalence, for that last reason.

   Nothing is guessed: the bindings are read off the unit itself. A
   variable that stands for a symbol is not looked through; following
   aliases is another pass (../types/rocq/FlambdaAliasesSpec.v), to be
   run before this one, and again after it for the symbols and constants
   that it puts in place of projections.

   What is not here: what a value slot holds, of a closure one of whose
   value slots holds a variable (a value computed when the unit runs),
   and what a field that is a variable holds; the variable is in scope
   where the symbol is bound and not in code. [Is_int] and [Get_tag] of
   a static block one of whose fields is a variable are not folded
   either.

   The proofs are in FlambdaKnownSymbolsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaNormaliseSpec FlambdaUnusedStaticsSpec FlambdaImportsSpec
  FlambdaObserversSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known                                                            *)
(* ======================================================================== *)

(* A symbol or a constant. *)
Definition closed_simple (s : simple) : bool :=
  match s with
  | Symbol _ | Const _ => true
  | Var _ => false
  end.

(* A static constant with no variable in it. *)
Definition closed_static_const (const : static_const) : bool :=
  match const with
  | Static_const_set_of_closures _ value_slots
  | Static_const_closure_of_set _ _ _ value_slots =>
    forallb (fun slot => closed_simple (snd slot)) value_slots
  | Static_const_block _ fields => forallb closed_simple fields
  | Static_const_immutable_value_array elements =>
    forallb closed_simple elements
  | Static_const_boxed_number contents => closed_simple contents
  | Static_const_immutable_string _ => true
  end.

(* What a binding lets be known: a symbol bound to a constant with no
   variable in it, which has a value. *)
Definition known_of_binding (binding : static_binding)
    : list (symbol * static_const) :=
  match binding with
  | Bind_symbol s const =>
    if closed_static_const const
    then
      match eval_static_const [] const with
      | Some _ => [(s, const)]
      | None => []
      end
    else []
  | Bind_code _ _ | Bind_deleted_code _ => []
  end.

(* The symbols that the expression binds to such constants, outside
   code, with the constants. *)
Fixpoint known_bindings (e : expr) : list (symbol * static_const) :=
  match e with
  | Let _ _ _ body => known_bindings body
  | Let_static bindings body =>
    flat_map known_of_binding bindings ++ known_bindings body
  | Let_cont_non_recursive _ h body =>
    known_bindings_of_handler h ++ known_bindings body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => known_bindings_of_handler h) handlers
    ++ known_bindings body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with known_bindings_of_handler (h : continuation_handler)
    : list (symbol * static_const) :=
  match h with
  | Handler _ _ handler => known_bindings handler
  end.

(* What each such symbol is bound to: the value of its constant, which
   needs no environment. There is no code. *)
Definition known_value (binding : symbol * static_const)
    : list (symbol * statically_allocated_value) :=
  match eval_static_const [] (snd binding) with
  | Some allocated => [(fst binding, allocated)]
  | None => []
  end.

Definition known_statics (known : list (symbol * static_const))
    : static_env :=
  Static_env [] (flat_map known_value known).

(* What stands, in what is known of a static block, for a field that is
   a variable: a constant that is not of kind [Value]. A load from a
   block of values has no value on such a field
   (FlambdaPrimitiveSpec.eval_block_load), so none is folded to it. *)
Definition unknown_field : const := Naked_immediate 0.

(* What is known of a closure of a static set whatever its value slots
   hold: the same closure with no value slot, which has a value if its
   function slot is one of the set. And of a static block whatever its
   fields that are variables hold: the same block with [unknown_field]
   for those. *)
Definition set_of_binding (binding : static_binding)
    : list (symbol * static_const) :=
  match binding with
  | Bind_symbol s (Static_const_set_of_closures code _) =>
    [(s, Static_const_set_of_closures code [])]
  | Bind_symbol s (Static_const_closure_of_set slot symbols decls _) =>
    let set := Static_const_closure_of_set slot symbols decls [] in
    match eval_static_const [] set with
    | Some _ => [(s, set)]
    | None => []
    end
  | Bind_symbol s (Static_const_block tag fields) =>
    [(s,
      Static_const_block tag
        (map
           (fun field =>
              if closed_simple field then field else Const unknown_field)
           fields))]
  | Bind_symbol _ _ | Bind_code _ _ | Bind_deleted_code _ => []
  end.

(* The symbols that the expression binds to closures of static sets and
   to static blocks, outside code, with that. *)
Fixpoint known_sets (e : expr) : list (symbol * static_const) :=
  match e with
  | Let _ _ _ body => known_sets body
  | Let_static bindings body =>
    flat_map set_of_binding bindings ++ known_sets body
  | Let_cont_non_recursive _ h body =>
    known_sets_of_handler h ++ known_sets body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => known_sets_of_handler h) handlers
    ++ known_sets body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with known_sets_of_handler (h : continuation_handler)
    : list (symbol * static_const) :=
  match h with
  | Handler _ _ handler => known_sets handler
  end.

(* Two closures at one function slot of one static set, whatever their
   value slots hold; for a set of one function with no name for its
   slot, two closures of one code. And two blocks of one tag and as
   many fields, the second with each field of the first that is not
   [unknown_field]. *)
Definition same_set (allocated allocated' : statically_allocated_value)
    : Prop :=
  match allocated, allocated' with
  | Static_closure code _, Static_closure code' _ => code' = code
  | Static_closure_of_set slot symbols decls _,
    Static_closure_of_set slot' symbols' decls' _ =>
    slot' = slot ∧ symbols' = symbols ∧ decls' = decls
  | Static_block tag fields, Static_block tag' fields' =>
    tag' = tag
    ∧ Forall2
        (fun known actual =>
           known = Value_const unknown_field ∨ actual = known)
        fields fields'
  | _, _ => False
  end.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* A constant, or a symbol that [statics] binds. *)
Definition simple_known (statics : static_env) (s : simple) : bool :=
  match s with
  | Const _ => true
  | Symbol sym => is_some (lookup sym (static_values statics))
  | Var _ => false
  end.

(* The load of bytes from a string, which Simplify does not compute
   even where the string and the index are known. *)
Definition reads_string (p : prim) : bool :=
  match p with
  | Binary (String_or_bigstring_load _) => true
  | Unary _ | Binary _ | Variadic _ | Heap _ | Region_primitive _ => false
  end.

(* A primitive that may be folded, or a projection, all of whose
   arguments are such. *)
Definition named_known (statics : static_env) (d : named) : bool :=
  match d with
  | Prim p args =>
    prim_may_fold p && negb (reads_string p)
    && forallb (simple_known statics) args
  | Project_value_slot _ closure | Project_value_slot_from _ _ closure
  | Project_function_slot _ _ closure =>
    simple_known statics closure
  | Simple _ | Set_of_closures _ _ | Closure_of_set _ _ _ => false
  end.

(* The defining expression, or its value under [statics] if all its
   arguments are known and the value is a constant or a symbol. *)
Definition fold_known_named (statics : static_env) (d : named) : named :=
  if named_known statics d
  then
    match eval_named statics [] d with
    | Some (Value_const c) => Simple (Const c)
    | Some (Value_symbol s) => Simple (Symbol s)
    | _ => d
    end
  else d.

(* The projection of a function slot from a symbol, or the symbol that
   it gives under [sets]: the evaluation of the machine reads the
   function slot, the functions and the symbols of the set, and no value
   slot. *)
Definition fold_function_slot (sets : static_env) (d : named) : named :=
  match d with
  | Project_function_slot _ _ (Symbol _) =>
    match eval_named sets [] d with
    | Some (Value_symbol s) => Simple (Symbol s)
    | _ => d
    end
  | _ => d
  end.

(* The load of a field from a symbol, where the load is not marked
   [Mutable], or the constant or the symbol that it gives under [sets]:
   the evaluation of the machine reads the tag, the number of fields
   and that field, and has no value on a field that is
   [unknown_field]. A load marked [Mutable] is left alone, as Simplify
   leaves one whose argument is not a constant
   (FlambdaBlockLoadSpec.v). *)
Definition fold_block_load (sets : static_env) (d : named) : named :=
  match d with
  | Prim (Unary (Block_load _ (Immutable | Immutable_unique) _)) [Symbol _] =>
    match eval_named sets [] d with
    | Some (Value_const c) => Simple (Const c)
    | Some (Value_symbol s) => Simple (Symbol s)
    | _ => d
    end
  | _ => d
  end.

(* An indirect call of a symbol that [sets] binds to a closure, made a
   direct call of the code of that closure. *)
Definition fold_known_apply (sets : static_env) (ap : apply_expr)
    : apply_expr :=
  match ap_call_kind ap, ap_callee ap with
  | Indirect, Symbol s =>
    match closure_of sets (Value_symbol s) with
    | Some (code, _) =>
      Apply_expr (ap_callee ap) (Direct code) (ap_args ap)
        (ap_continuation ap) (ap_exn_continuation ap) (ap_exn_extra_args ap)
        (ap_args_arity ap) (ap_return_arity ap)
    | None => ap
    end
  | Indirect, (Var _ | Const _) | Direct _, _ => ap
  end.

(* Every [Let] of the expression, in handlers and in code too, with its
   defining expression folded: by what [statics] binds, and then, if it
   is still the projection of a function slot or the load of a field, by
   what [sets] binds. And every call, with its kind folded by what
   [sets] binds. *)
Fixpoint fold_known (statics sets : static_env) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind
      (fold_block_load sets
         (fold_function_slot sets (fold_known_named statics d)))
      (fold_known statics sets body)
  | Let_static bindings body =>
    Let_static (map (fold_known_binding statics sets) bindings)
      (fold_known statics sets body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (fold_known_handler statics sets h)
      (fold_known statics sets body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, fold_known_handler statics sets h)) handlers)
      (fold_known statics sets body)
  | Apply ap => Apply (fold_known_apply sets ap)
  | Apply_cont _ | Switch _ _ | Invalid => e
  end
with fold_known_handler (statics sets : static_env)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (fold_known statics sets handler)
  end
with fold_known_binding (statics sets : static_env)
    (binding : static_binding) : static_binding :=
  match binding with
  | Bind_code id c => Bind_code id (fold_known_code statics sets c)
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end
with fold_known_code (statics sets : static_env) (c : code) : code :=
  match c with
  | Code params my_closure ret exn metadata body =>
    Code params my_closure ret exn metadata (fold_known statics sets body)
  end.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

(* If [known] gives the symbol a constant, the binding binds it to that
   constant, which has no variable in it; and if [sets] gives it a
   closure of a set, the binding binds it to a closure at that slot of
   that set, and if a block, to a block that is the same block once
   each field that is a variable is replaced by [unknown_field]: one
   of that tag, with as many fields, and with the same field wherever
   neither has a variable. The test is the equality of the two blocks
   so replaced, so a variable in one may stand where the other has a
   variable or the constant [unknown_field] itself; a load of that
   field from the block so replaced has no value, so none is folded
   to it. *)
Definition binding_bound_as (known sets : list (symbol * static_const))
    (binding : static_binding) : bool :=
  match binding with
  | Bind_symbol s const =>
    match lookup s known with
    | Some expected =>
      closed_static_const const
      && if static_const_eq_dec const expected then true else false
    | None => true
    end
    && match lookup s sets with
       | Some expected =>
         match set_of_binding binding with
         | [(_, set)] =>
           if static_const_eq_dec set expected then true else false
         | _ => false
         end
       | None => true
       end
  | Bind_code _ _ | Bind_deleted_code _ => true
  end.

(* Every static binding of the expression, in handlers and in code too,
   binds a symbol of [known] to the constant that [known] gives it, and
   one of [sets] to a closure of the set that [sets] gives it, or to a
   block that is, with its fields that are variables replaced, the
   block that [sets] gives it. *)
Fixpoint bound_as (known sets : list (symbol * static_const)) (e : expr)
    : bool :=
  match e with
  | Let _ _ _ body => bound_as known sets body
  | Let_static bindings body =>
    forallb (binding_bound_as known sets) bindings
    && forallb (code_of_binding_bound_as known sets) bindings
    && bound_as known sets body
  | Let_cont_non_recursive _ h body =>
    handler_bound_as known sets h && bound_as known sets body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_bound_as known sets h) handlers
    && bound_as known sets body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_bound_as (known sets : list (symbol * static_const))
    (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => bound_as known sets handler
  end
with code_of_binding_bound_as (known sets : list (symbol * static_const))
    (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_bound_as known sets c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_bound_as (known sets : list (symbol * static_const)) (c : code)
    : bool :=
  match c with
  | Code _ _ _ _ _ body => bound_as known sets body
  end.

(* The unit with what it binds its symbols to used wherever they are, if
   it binds each of them to one constant, each symbol of a closure of a
   set to closures at one slot of one set, and each symbol of a block
   to blocks that are the same once each field that is a variable is
   replaced by [unknown_field]. *)
Definition fold_known_unit (e : expr) : option expr :=
  let known := known_bindings e in
  let sets := known_sets e in
  if bound_as known sets e
  then Some (fold_known (known_statics known) (known_statics sets) e)
  else None.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* What [fold_known_unit] gives refines the unit, run on its own. *)
Definition fold_known_unit_program_refines_statement : Prop :=
  ∀ e e' : expr, fold_known_unit e = Some e' -> program_refines e e'.

(* The same of a unit that imports symbols, run from the static
   environment of its imports: if the unit binds no name that the
   imports bind, the result refines it. The rewrite puts, for a symbol,
   what the unit binds it to, wherever the symbol is used: also before
   the binding is executed, where a symbol that is an import still has
   what the import is. *)
Definition fold_known_unit_refines_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (e e' : expr),
    fold_known_unit e = Some e' ->
    static_names_free_in (imports_statics imports) e = true ->
    unit_refines imports e e'.

(* The same of the unit followed by something: the unit and the unit
   folded, run from one continuation environment of what may follow a
   unit (FlambdaObserversSpec.observers: a way out, or a handler with no
   environment of its own that binds nothing static and names no
   symbol and no code id) and from the static environment of the
   imports. The second refines the first, with the same exits: what
   follows is the same term in the two runs, it is not folded, and it
   reads the same values. With no continuation it is the statement
   above. *)
Definition fold_known_unit_refines_observed_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (conts : cont_env)
    (e e' : expr),
    fold_known_unit e = Some e' ->
    static_names_free_in (imports_statics imports) e = true ->
    observers [] conts ->
    refines (State e [] conts (imports_statics imports) empty_heap)
      (State e' [] conts (imports_statics imports) empty_heap).

(* A defining expression is folded to its value: in any static
   environment in which each symbol that [statics] binds is bound to the
   same thing or to nothing, if the expression has a value, the
   expression folded has that value. *)
Definition fold_known_named_sound_statement : Prop :=
  ∀ (statics run_statics : static_env) (vars : env) (d : named) (v : value),
    (∀ (s : symbol) (allocated allocated' : statically_allocated_value),
      lookup s (static_values statics) = Some allocated ->
      lookup s (static_values run_statics) = Some allocated' ->
      allocated' = allocated) ->
    eval_named run_statics vars d = Some v ->
    eval_named run_statics vars (fold_known_named statics d) = Some v.

(* The projection of a function slot is folded to its value: in any
   static environment in which each symbol that [sets] binds is bound
   to a closure at the same slot of the same set, to a block of the
   same tag with the fields that are known, or to nothing, if the
   expression has a value, the expression folded has that value. *)
Definition fold_function_slot_sound_statement : Prop :=
  ∀ (sets run_statics : static_env) (vars : env) (d : named) (v : value),
    (∀ (s : symbol) (allocated allocated' : statically_allocated_value),
      lookup s (static_values sets) = Some allocated ->
      lookup s (static_values run_statics) = Some allocated' ->
      same_set allocated allocated') ->
    eval_named run_statics vars d = Some v ->
    eval_named run_statics vars (fold_function_slot sets d) = Some v.

(* The load of a field is folded to its value, in any such static
   environment. *)
Definition fold_block_load_sound_statement : Prop :=
  ∀ (sets run_statics : static_env) (vars : env) (d : named) (v : value),
    (∀ (s : symbol) (allocated allocated' : statically_allocated_value),
      lookup s (static_values sets) = Some allocated ->
      lookup s (static_values run_statics) = Some allocated' ->
      same_set allocated allocated') ->
    eval_named run_statics vars d = Some v ->
    eval_named run_statics vars (fold_block_load sets d) = Some v.

(* A call is folded to a call that steps as it does, in any such static
   environment. *)
Definition fold_known_apply_sound_statement : Prop :=
  ∀ (sets run_statics : static_env) (hp : heap) (vars : env) (conts : cont_env)
    (ap : apply_expr),
    (∀ (s : symbol) (allocated allocated' : statically_allocated_value),
      lookup s (static_values sets) = Some allocated ->
      lookup s (static_values run_statics) = Some allocated' ->
      same_set allocated allocated') ->
    step (State (Apply (fold_known_apply sets ap)) vars conts run_statics hp)
    = step (State (Apply ap) vars conts run_statics hp).

(* The rewrite commutes with renaming, which renames no symbol: what is
   known is the same, a defining expression that is folded has no
   variable in it, and whether a call is folded is read off its kind and
   a callee that is a symbol, with its arguments left as they are. *)
Definition fold_known_equivariant_statement : Prop :=
  ∀ (r : renaming) (statics sets : static_env) (e : expr),
    rename_expr r (fold_known statics sets e)
    = fold_known statics sets (rename_expr r e).
