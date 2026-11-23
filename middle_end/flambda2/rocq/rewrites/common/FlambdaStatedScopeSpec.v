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

(* What is known, beyond its kind, of the value of a variable bound
   around a point of a term.

   This file is to be reviewed: the definitions and the statements at
   the end. The tests are asked by the rewrites of the validation that
   take away the application of a handler or a call (the inlining of
   a handler at a tail, the inlining of a known call, the two
   forwarders, the rewrites of a parameter that every application
   gives one thing), which did nothing where the binder states
   anything (FlambdaKindModel.subkinds_say_nothing). The proofs are in
   FlambdaStatedScopeProofs.v, which needs no review.

   The machine has no step where a handler is applied, or the body of
   code entered, with a value that is not of what the parameter states
   (FlambdaSubkindSpec.values_of_subkinds; decision D25, at
   parameters). A rewrite of the term after Simplify that takes an
   application away takes that test away. That the test cannot fail,
   each argument being known to be of what the parameter states of it,
   is a sufficient reason to take it away, and is the one asked here;
   it is not said to be the weakest. The rest of the equivalence is
   the rewrite's own proof.

   What is known of a variable comes from the term itself, going down
   it ([stated_in_scope], the most recently bound first, as the
   variables of the machine are):
   - a parameter of a handler holds a value of what the handler states
     of it, since the machine tested it where the handler was applied
     ([params_stated]);
   - a variable bound by a [Let] holds what its defining expression
     makes ([stated_of_named]): a constant, a symbol, another variable,
     a tagged immediate from the primitives that make one, a boxed
     number from [Box_number], a block from [Make_block] with what is
     known of its fields, and a field loaded from a variable known to
     be a block of one shape, with what that shape says of the field
     (FlambdaSubkindBinderSpec.load_of_block_subkind_statement); a
     tagged immediate from a load whose access kind says that the field
     is an immediate, and from a load of an array at [Load_immediates],
     neither of which has a value on anything else
     (FlambdaPrimitiveSpec.eval_block_load, array_loaded); what
     [Opaque_identity] is given; the length of an array, a tagged
     immediate; and a closure, an immutable array and a block or an
     array with an identity, none of which is null.
   Of anything else nothing is known ([any_value]).

   [are_stated] is the test: each argument is known to be of what the
   list states at its position. For a constant it is the test of the
   machine itself, which reads no static environment for a constant;
   for a symbol and a variable it is the order of
   FlambdaSubkindOrderSpec.v between what is known and what is
   stated.

   The invariant ([stated_hold]). A scope holds in an environment of
   the machine where each variable of the scope that the environment
   binds is bound to a value that the machine's own test accepts at
   what the scope says (FlambdaSubkindSpec.value_at). The statements
   say that the machine keeps it, binder by binder, in step with what
   the rewrites add to the scope:
   - a [Let]: the step binds the variable to a value of what
     [stated_of_named] says ([stated_of_named_sound_statement],
     [stated_hold_let_statement]);
   - a handler that is applied: the step binds the parameters, in the
     environment in which the handler was bound, to values of what
     the handler states, as the machine has just tested them
     ([params_stated_hold_statement]);
   - the body of code that is entered: the same of what the code
     states of its parameters ([code_stated_hold_statement]);
   - a [Let_static]: the static environment grows and what was bound
     stays bound to the same (FlambdaWellFormedSpec.
     step_extends_statics_statement), and what a value is of, it is
     still of ([stated_hold_extend_statement]).
   Nothing else binds a variable. No step changes a binding that is
   there: a new binder may bind a variable again, and then the scope
   and the environment each get the new entry in front, together, so
   that what was known of the variable bound before is no longer read.
   The test reads no heap, so a store changes nothing that is known; a
   value that is a location is known as not null at most.
   And where the invariant holds, the test of a rewrite is the test of
   the machine: if [are_stated] accepts the arguments at a list, the
   machine accepts their values at that list
   ([are_stated_sound_statement]). So an application that a rewrite
   takes away is one whose test could not fail.

   What each line of [stated_of_named] rests on, in the machine:
   - [Simple]: the value of the [Simple] ([stated_of_simple]: a
     variable by the invariant; a symbol is a value that is neither
     null nor poison; a tagged immediate in range is the constant
     constructor of that number);
   - [Tag_immediate], [Boolean_not], the arithmetic and the shifts at
     [Standard_tagged_immediate], [Num_conv] to it, [Array_length]:
     FlambdaPrimitiveSpec.eval_prim makes a tagged immediate in range
     or has no value (the section "Numbers in range" there);
   - [Box_number]: it makes the boxed number of that kind;
   - [Make_block]: it has no value unless each argument is of kind
     [Value], and makes the block of that tag with the values of the
     arguments, each of which is of what is known of its [Simple];
   - [Block_load] of a variable known as a block of one shape:
     FlambdaSubkindBinderSpec.load_of_block_subkind_statement; with
     the field kind [Immediate], the load has no value on a field
     that is not a tagged immediate (eval_block_load, and
     FlambdaHeapSpec.load_field for a block with an identity);
   - [Array_load] at [Load_immediates]: array_loaded has no value on
     an element that is not a tagged immediate, for an immutable
     array and for one with an identity; at the two other load kinds
     nothing is known of the element, not even that it is not null;
   - [Opaque_identity]: its value is its argument;
   - closures, [Project_function_slot], [Make_immutable_array],
     [Make_block_with_identity], [Make_mutable_array]: the value is a
     closure, a symbol that the projection of a function slot gives
     back (which need not be bound to a closure, nor bound at all), an
     array or a location, none of which is null or poison. Not null
     is all that is said of it: not that it is of kind [Value], that
     a symbol is bound or that a location is live.

   Not covered, and why it matters to a reader: the statements are of
   single steps. That a rewrite which threads the scope down a term
   meets, at each node, a state of the machine in which the scope
   holds is part of the proof of that rewrite (its simulation carries
   [stated_hold], as FlambdaNormaliseProofs.kinds_hold is carried for
   the kinds), and is not stated here. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveSpec FlambdaSubkindSpec
  FlambdaSubkindOrderSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaSubkindBinderSpec.
Import ListNotations.

Definition stated_in_scope : Type := list (variable * param_subkind).

Definition stated_of_var (S : stated_in_scope) (x : variable)
    : param_subkind :=
  match lookup x S with
  | Some ps => ps
  | None => any_value
  end.

Definition not_null : param_subkind := (false, Subkind_anything).

Definition tagged : param_subkind := (false, Subkind_tagged_immediate).

(* What is known of the value of a [Simple]. A tagged immediate is
   known as the variant of that one constant constructor, which says
   as much as a tagged immediate and as any variant that has it. *)
Definition stated_of_simple (S : stated_in_scope) (s : simple)
    : param_subkind :=
  match s with
  | Var x => stated_of_var S x
  | Symbol _ => not_null
  | Const (Tagged_immediate i) =>
    if is_tagged_immediate (Value_const (Tagged_immediate i))
    then (false, Subkind_variant [i] Shapes_nil)
    else any_value
  | Const _ => any_value
  end.

(* What is known of the fields of a block made of these. *)
Fixpoint fields_stated (S : stated_in_scope) (args : list simple)
    : field_subkinds :=
  match args with
  | [] => Fields_nil
  | s :: args =>
    let ps := stated_of_simple S s in
    Fields_cons (fst ps) (snd ps) (fields_stated S args)
  end.

(* What a value known to be of this subkind has at a field, where the
   subkind is of blocks of one shape: no other tag, and the fields
   given. The constant constructors do not matter: a load from one has
   no value. *)
Definition stated_of_field (ps : param_subkind) (field : Z)
    : param_subkind :=
  match snd ps with
  | Subkind_variant _ (Shape_determined _ fields Shapes_nil) =>
    if Z.leb 0 field
    then
      match field_subkind_at fields (Z.to_nat field) with
      | Some ps' => ps'
      | None => any_value
      end
    else any_value
  | _ => any_value
  end.

(* What the value of a defining expression is known to be, where it
   has one. *)
Definition stated_of_named (S : stated_in_scope) (d : named)
    : param_subkind :=
  match d with
  | Simple s => stated_of_simple S s
  | Prim (Unary Tag_immediate) _ => tagged
  | Prim (Unary Boolean_not) _ => tagged
  | Prim (Unary (Unary_int_arith Standard_tagged_immediate _)) _ => tagged
  | Prim (Binary (Int_arith Standard_tagged_immediate _)) _ => tagged
  | Prim (Binary (Int_shift Standard_tagged_immediate _)) _ => tagged
  | Prim (Unary (Box_number n)) _ => (false, Subkind_boxed_number n)
  | Prim (Variadic (Make_block (Block_kind_values tag))) args =>
    (false,
     Subkind_variant []
       (Shape_determined tag (fields_stated S args) Shapes_nil))
  | Prim (Unary (Block_load (Block_access_values _ _ field_kind) _ field))
      args =>
    let ps :=
      match args with
      | [Var b] => stated_of_field (stated_of_var S b) field
      | _ => any_value
      end
    in
    match field_kind with
    | Immediate => if param_says_at_least ps tagged then ps else tagged
    | Any_value => ps
    end
  | Prim (Binary (Array_load _ Load_immediates _)) _ => tagged
  | Prim (Unary (Opaque_identity _)) [s] => stated_of_simple S s
  | Prim (Unary (Array_length _)) _ => tagged
  | Prim (Unary (Num_conv _ Standard_tagged_immediate)) _ => tagged
  | Prim (Variadic (Make_immutable_array _)) _ => not_null
  | Prim (Heap (Make_block_with_identity _ _)) _ => not_null
  | Prim (Heap (Make_mutable_array _)) _ => not_null
  | Set_of_closures _ _ => not_null
  | Closure_of_set _ _ _ => not_null
  | Project_function_slot _ _ _ => not_null
  | _ => any_value
  end.

(* The parameters of a binder with what it states of them, the last
   parameter first; a parameter with no entry is known as nothing. *)
Fixpoint params_stated (params : list variable) (sks : list param_subkind)
    (S : stated_in_scope) : stated_in_scope :=
  match params with
  | [] => S
  | param :: params =>
    match sks with
    | ps :: sks => params_stated params sks ((param, ps) :: S)
    | [] => params_stated params [] ((param, any_value) :: S)
    end
  end.

(* Whether a [Simple] is known to be of what a binder states. *)
Definition is_stated (S : stated_in_scope) (ps : param_subkind) (s : simple)
    : bool :=
  match s with
  | Const c => value_at empty_static_env ps (Value_const c)
  | Var _ | Symbol _ => param_says_at_least (stated_of_simple S s) ps
  end.

(* Whether the arguments are known to be of what is stated, one for
   one. As FlambdaSubkindSpec.values_of_subkinds, it stops where either
   list ends: nothing is asked of the arguments that are left where
   the list of what is stated is the shorter, nor of what is stated
   beyond the arguments. It does not say that there are as many
   arguments as parameters, nor that an argument has a value. *)
Fixpoint are_stated (S : stated_in_scope) (args : list simple)
    (sks : list param_subkind) {struct sks} : bool :=
  match sks, args with
  | ps :: sks, s :: args => is_stated S ps s && are_stated S args sks
  | _, _ => true
  end.

(* What a call binds, with what the code states of its parameters: the
   machine tests them where the body is entered
   (FlambdaSemanticsSpec.call_code). The closure called is no null and
   no poison. *)
Definition code_stated (params : list variable) (my_closure : variable)
    (metadata : code_metadata) : stated_in_scope :=
  params_stated params (cm_params_subkinds metadata)
    [(my_closure, not_null)].

(* Whether a [Simple] is known to be of what the list states at a
   position; nothing is asked where the list has no entry there. *)
Definition is_stated_at (S : stated_in_scope) (sks : list param_subkind)
    (i : nat) (s : simple) : bool :=
  match nth_error sks i with
  | Some ps => is_stated S ps s
  | None => true
  end.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* What is known holds in an environment: a variable of the scope that
   the environment binds is bound to a value of what the scope says of
   it. Both lists are read as the machine reads its variables: of two
   entries for one variable the one bound last counts. *)
Definition stated_hold (statics : static_env) (S : stated_in_scope)
    (vars : env) : Prop :=
  ∀ (x : variable) (ps : param_subkind) (v : value),
    lookup x S = Some ps ->
    lookup x vars = Some v ->
    value_at statics ps v = true.

(* ---- The test of the machine, of one value ----------------------------- *)

(* Every value is of what says nothing. *)
Definition value_at_any_value_statement : Prop :=
  ∀ (statics : static_env) (v : value),
    value_at statics any_value v = true.

(* The test of a constant reads no static environment. *)
Definition value_at_const_statement : Prop :=
  ∀ (statics : static_env) (ps : param_subkind) (c : const),
    value_at statics ps (Value_const c)
    = value_at empty_static_env ps (Value_const c).

(* FlambdaSubkindOrderSpec.values_of_subkinds_weaken_statement, of one
   value. *)
Definition value_at_weaken_statement : Prop :=
  ∀ (statics : static_env) (a b : param_subkind) (v : value),
    param_says_at_least a b = true ->
    value_at statics a v = true ->
    value_at statics b v = true.

(* FlambdaSubkindBinderSpec.values_of_subkinds_extend_statement, of one
   value. *)
Definition value_at_extend_statement : Prop :=
  ∀ (statics statics' : static_env) (ps : param_subkind) (v : value),
    statics_extend statics statics' ->
    value_at statics ps v = true ->
    value_at statics' ps v = true.

(* ---- The machine keeps what is known ----------------------------------- *)

Definition stated_hold_extend_statement : Prop :=
  ∀ (statics statics' : static_env) (S : stated_in_scope) (vars : env),
    statics_extend statics statics' ->
    stated_hold statics S vars ->
    stated_hold statics' S vars.

Definition stated_of_simple_sound_statement : Prop :=
  ∀ (statics : static_env) (S : stated_in_scope) (vars : env) (s : simple)
    (v : value),
    stated_hold statics S vars ->
    eval_simple vars s = Some v ->
    value_at statics (stated_of_simple S s) v = true.

(* The value that the machine gives a defining expression is of what
   [stated_of_named] says, whatever the heap. *)
Definition stated_of_named_sound_statement : Prop :=
  ∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
    (S : stated_in_scope) (vars : env) (d : named) (v : value),
    stated_hold statics S vars ->
    exec_named statics hp rd vars d = Some (v, hp', rd') ->
    value_at statics (stated_of_named S d) v = true.

(* So the step of a [Let] keeps what is known, with the entry that the
   rewrites add. *)
Definition stated_hold_let_statement : Prop :=
  ∀ (S : stated_in_scope) (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record) (s : state),
    stated_hold statics S vars ->
    step (State (Let x kind d body) vars conts statics hp rd) = Step s ->
    stated_hold (state_statics s) ((x, stated_of_named S d) :: S)
      (state_vars s).

(* A handler that is applied runs with its parameters bound to values
   of what it states, around what was known where it was bound. *)
Definition params_stated_hold_statement : Prop :=
  ∀ (S : stated_in_scope) (params : list variable)
    (metadata : handler_metadata) (handler : expr) (def_vars : env)
    (def_conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (vs : list value) (statics : static_env) (hp : heap) (rd : read_record)
    (s : state),
    stated_hold statics S def_vars ->
    apply_cont_value
      (Closure (Handler params metadata handler) def_vars def_conts recursive)
      vs statics hp rd
    = Step s ->
    stated_hold (state_statics s)
      (params_stated params (hm_params_subkinds metadata) S) (state_vars s).

(* The body of code that is entered runs with its parameters bound to
   values of what the code states. The premise on the closure is what
   [code_stated] says of [my_closure]. [call_code] does not test it, and
   this statement does not say that a call that has a step gives it. The
   machine enters code with the value in which it found the code
   (FlambdaSemanticsSpec.shape_of_call, which has read
   FlambdaValueSpec.closure_of of it, and that has no value on null or
   on poison): a proof that uses this statement owes the premise from
   there. Nothing is said of what the code states of its results. *)
Definition code_stated_hold_statement : Prop :=
  ∀ (closure : value) (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata) (body : expr)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
    (rd : read_record) (s : state),
    value_at statics not_null closure = true ->
    call_code closure (Code params my_closure ret exn metadata body) vs
      args_arity return_arity return_to exn_to statics hp rd
    = Step s ->
    stated_hold (state_statics s) (code_stated params my_closure metadata)
      (state_vars s).

(* ---- The test of a rewrite is the test of the machine ------------------ *)

Definition is_stated_sound_statement : Prop :=
  ∀ (statics : static_env) (S : stated_in_scope) (vars : env)
    (ps : param_subkind) (s : simple) (v : value),
    stated_hold statics S vars ->
    is_stated S ps s = true ->
    eval_simple vars s = Some v ->
    value_at statics ps v = true.

Definition are_stated_sound_statement : Prop :=
  ∀ (statics : static_env) (S : stated_in_scope) (vars : env)
    (args : list simple) (sks : list param_subkind) (vs : list value),
    stated_hold statics S vars ->
    are_stated S args sks = true ->
    eval_simples vars args = Some vs ->
    values_of_subkinds statics vs sks = true.

Definition is_stated_at_sound_statement : Prop :=
  ∀ (statics : static_env) (S : stated_in_scope) (vars : env)
    (sks : list param_subkind) (i : nat) (ps : param_subkind) (s : simple)
    (v : value),
    stated_hold statics S vars ->
    is_stated_at S sks i s = true ->
    nth_error sks i = Some ps ->
    eval_simple vars s = Some v ->
    value_at statics ps v = true.
