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

(* The proofs of FlambdaReaperUnboxingRefinesSpec.v. Nothing here needs
   review.

   The rebuild with unboxing is compared with the plain rebuild, which
   FlambdaReaperRebuildProofs.v compares with the unit. The two rebuilt terms
   are related by [ub]: equal but at the [Let] of an unboxed variable, which the
   unboxing makes [Let]s of the variables of its fields ([lets]: of what the
   allocation holds, of the variables of the fields of the variable it is an
   alias of, or, for a field unboxed in turn, of those of the variable that the
   allocation holds there: [alloc_entry]), at a read of a field of it, which it
   makes the variable of the field, at a read of a field unboxed in turn into an
   unboxed variable ([ub_load]), at a handler, whose unboxed parameters it makes
   the variables of their fields ([params_rel]), and at an [Apply_cont] to such
   a handler, whose unboxed arguments it makes the variables of their fields
   that the parameter has ([args_rel]). [ub] is indexed by the shapes of the
   parameters of the continuations in scope (which are unboxed, with which
   fields); those of continuations not free in a term do not matter
   ([ub_agree]). Related terms run alike ([ub_refines]), by a simulation of
   FlambdaReaperRebuildProofs.v ([shrinking_refines]): a step of the plain term
   is answered by as many steps as the unboxing makes [Let]s, none where it
   makes none; an unboxed variable bound in the plain run to a value has the
   variables of its fields bound in the other to the fields of the value, and
   those of the fields of a field unboxed in turn to the fields of that field
   ([env_rel]), so that the arguments of an [Apply_cont] related are bound to
   parameters related ([bind_args_rel]). Immutable blocks and boxed numbers are
   values, so no heap is involved.

   The two rebuilds of a unit of the fragment are related ([rebuild_ub]),
   the plain rebuild with the masks of the handlers in scope and the
   rebuild with unboxing with their decisions, which the fragment's view
   of them ([conts_match]) relates. What the plain rebuild keeps of an
   allocation that is unboxed, it keeps because the solution is closed
   ([field_kept]): each field unboxed is read with a use from a use of
   the variable ([get_fields_in] and the lemmas of "Where the unboxed
   fields come from"), the graph of the unit reads no function slot and
   makes no call ([closure_free_reads_blocks]) so that each such use is a
   use of the variable, and the facts of the allocation are those of
   FlambdaReaperFlowProofs.expr_edges. The variables of the fields are
   fresh ([unit_unboxed_fields_fresh]), so that the checks of the
   fragment hold with "plain" read as "neither unboxed nor the variable
   of a field" ([strong_local]), which is what the relation asks. The
   statements compose. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaBindersSpec FlambdaWellFormedSpec
  FlambdaNameAbstractionSpec FlambdaRenameApartSpec FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaUnusedStaticsSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionProofs FlambdaReaperRebuild
  FlambdaReaperFlowSpec FlambdaReaperRebuildSpec FlambdaReaperRebuildProofs
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs.
Import ListNotations.

(* ======================================================================== *)
(* The fragment, with the names of the fields fresh                         *)
(* ======================================================================== *)

(* The fragment of FlambdaReaperUnboxingRefinesSpec.v asks that unboxed
   variables be only read; the proofs use too that the variables of the
   fields are no variable of the unit, which the model makes them
   ([unit_unboxed_fields_fresh], below): the same checks, a variable or a
   simple being [plain] if it is not unboxed and not the variable of a
   field. *)
Section Fragment.

Variable unboxed : list (node * unboxed_fields).
Variable sol : solution.

(* The variables that stand for the unboxed fields. *)
Definition field_variables : list variable :=
  flat_map (fun '(_, fields) => map fst (fields_leaves fields)) unboxed.

Definition plain_variable (x : variable) : bool :=
  negb (Keyed.is_bound node_eqb (Node_variable x) unboxed)
  && negb (among x field_variables).

Definition plain_simple (s : simple) : bool :=
  match s with
  | Var x => plain_variable x
  | Symbol sym => negb (Keyed.is_bound node_eqb (Node_symbol sym) unboxed)
  | Const _ => true
  end.

Definition unboxed_binding_ok (kinds : finite_map flambda_kind)
    (x : variable) (kind : flambda_kind) (d : named) (fields : unboxed_fields)
    : bool :=
  match kind, d with
  | Value, Prim (Variadic (Make_block (Block_kind_values _))) args =>
    forallb
      (fun '(f, u) =>
         match f, u with
         | Field_block i Value, Not_unboxed _ =>
           match nth_error args i with
           | Some arg =>
             plain_simple arg
             && kind_eqb (simple_kind (kinds_env kinds) arg) Value
           | None => false
           end
         | Field_block i Value, Unboxed nested =>
           match nth_error args i with
           | Some (Var b) =>
             negb (Pos.eqb x b)
             && kind_eqb (simple_kind (kinds_env kinds) (Var b)) Value
             && unboxed_arg_ok unboxed sol kinds true nested (Var b)
           | _ => false
           end
         | _, _ => false
         end)
      fields
  | Value, Prim (Unary (Box_number bn)) [contents] =>
    plain_simple contents
    && forallb
         (fun '(f, u) =>
            match u with
            | Not_unboxed _ => field_eqb f (Field_boxed_number bn)
            | Unboxed _ => false
            end)
         fields
  | Value, Simple (Var y) =>
    negb (Pos.eqb x y) && unboxed_arg_ok unboxed sol kinds true fields (Var y)
  | Value, Prim (Unary (Block_load _ _ i)) [Var y] =>
    negb (Pos.eqb x y)
    && match fields_of unboxed y with
       | Some fields_y =>
         match Keyed.lookup field_eqb (Field_block (Z.to_nat i) Value) fields_y
         with
         | Some (Unboxed nested) => fields_alike fields nested
         | _ => false
         end
       | None => false
       end
  | _, _ => false
  end.

Definition plain_binding_ok (x : variable) (kind : flambda_kind) (d : named)
    : bool :=
  match unboxed_read unboxed d with
  | Some (f, fields) =>
    negb (is_var_used sol x kind)
    || match Keyed.lookup field_eqb f fields with
       | Some (Not_unboxed _) => true
       | _ => false
       end
  | None =>
    match d with
    | Simple s => plain_simple s
    | Prim _ args => forallb plain_simple args
    | _ => false
    end
  end.

Definition local_binding (binding : static_binding) : bool :=
  match binding with
  | Bind_symbol _ (Static_const_block _ fields) =>
    forallb plain_simple fields
  | _ => false
  end.

Fixpoint args_ok (kinds : finite_map flambda_kind)
    (params : list (variable * bool)) (args : list simple) : bool :=
  match params, args with
  | _, [] => true
  | [], _ :: _ => forallb plain_simple args
  | (p, keep) :: params, a :: args =>
    match fields_of unboxed p with
    | Some fields => unboxed_arg_ok unboxed sol kinds keep fields a
    | None => negb keep || plain_simple a
    end
    && args_ok kinds params args
  end.

Definition apply_cont_ok (kinds : finite_map flambda_kind)
    (conts : finite_map (list (variable * bool))) (ac : apply_cont) : bool :=
  match lookup (ac_continuation ac) conts with
  | Some params => args_ok kinds params (ac_args ac)
  | None => forallb plain_simple (ac_args ac)
  end.

Definition param_ok (p : variable) (kind : flambda_kind) (keep : bool)
    : bool :=
  negb (among p field_variables)
  && FlambdaReaperUnboxingRefinesSpec.param_ok unboxed p kind keep.

Fixpoint local_expr (kinds : finite_map flambda_kind)
    (conts : finite_map (list (variable * bool))) (e : expr) : bool :=
  match e with
  | Let x kind d body =>
    negb (among x field_variables)
    && match fields_of unboxed x with
       | Some fields => unboxed_binding_ok kinds x kind d fields
       | None => plain_binding_ok x kind d
       end
    && local_expr ((x, kind) :: kinds) conts body
  | Let_static bindings body =>
    forallb local_binding bindings && local_expr kinds conts body
  | Let_cont_non_recursive k h body =>
    let mask := handler_mask sol (handler_is_exn_handler h) h in
    local_handler kinds conts mask h
    && local_expr kinds ((k, combine (handler_variables h) mask) :: conts)
         body
  | Let_cont_recursive handlers body =>
    let conts :=
      map
        (fun '(k, h) =>
           (k, combine (handler_variables h) (handler_mask sol false h)))
        handlers
      ++ conts
    in
    forallb
      (fun '(_, h) => local_handler kinds conts (handler_mask sol false h) h)
      handlers
    && local_expr kinds conts body
  | Apply _ => false
  | Apply_cont ac => apply_cont_ok kinds conts ac
  | Switch scrutinee arms =>
    plain_simple scrutinee
    && forallb (fun '(_, ac) => apply_cont_ok kinds conts ac) arms
  | Invalid => true
  end
with local_handler (kinds : finite_map flambda_kind)
    (conts : finite_map (list (variable * bool))) (mask : list bool)
    (h : continuation_handler) : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb (fun '((p, kind), keep) => param_ok p kind keep)
         (combine (combine params (hm_params_kinds metadata)) mask)
    && local_expr (rev (combine params (hm_params_kinds metadata)) ++ kinds)
         conts handler
  end.

End Fragment.

(* ======================================================================== *)
(* Terms related by local unboxing                                          *)
(* ======================================================================== *)

Section Related.

(* The unboxed fields of the names; their variables are distinct. *)
Variable M : list (node * unboxed_fields).
Hypothesis M_distinct : distinct (field_variables M) = true.

(* [Let]s of simples, the first the innermost. *)
Definition lets (l : list (variable * flambda_kind * simple)) (body : expr)
    : expr :=
  List.fold_left (fun body '(v, kind, s) => Let v kind (Simple s) body) l body.

(* The variable of the field [f] of the allocation [d] is bound to [s], at
   [kind]. *)
Definition alloc_field (d : named) (f : field) (kind : flambda_kind)
    (s : simple) : Prop :=
  match d with
  | Prim (Variadic (Make_block (Block_kind_values _))) args =>
    ∃ i : nat,
      f = Field_block i Value ∧ kind = Value ∧ nth_error args i = Some s
  | Prim (Unary (Box_number bn)) [c] =>
    f = Field_boxed_number bn ∧ kind = unboxed_kind bn ∧ s = c
  | _ => False
  end.

(* The field [f] of a value. *)
Definition field_value (f : field) (w : value) : option value :=
  match f, w with
  | Field_block i _, Value_block _ vs => nth_error vs i
  | Field_boxed_number _, Value_boxed_number c => Some (Value_const c)
  | _, _ => None
  end.

(* The read of a field: [Block_load] or [Unbox_number]. *)
Definition read_of (d : named) : option (field * simple) :=
  match d with
  | Prim (Unary (Block_load _ _ _)) [_] | Prim (Unary (Unbox_number _)) [_] =>
    field_read d
  | _ => None
  end.

Definition named_plain (d : named) : bool :=
  match d with
  | Simple s => plain_simple M s
  | Prim _ args => forallb (plain_simple M) args
  | _ => false
  end.

(* What the unboxing binds for the field [f] of the allocation [d] of
   [x]: the variable of the field to what [d] holds there, a simple of
   [plain_simple]; or, where the field is unboxed in turn, the variables
   of its fields to those of the variable that [d] holds there, another
   unboxed variable. *)
Definition alloc_entry (x : variable) (d : named)
    (fu : field * unboxed_field)
    (l : list (variable * flambda_kind * simple)) : Prop :=
  match fu with
  | (f, Not_unboxed v) =>
    ∃ (kind : flambda_kind) (s : simple),
      l = [(v, kind, s)] ∧ alloc_field d f kind s ∧ plain_simple M s = true
  | (f, Unboxed nested) =>
    ∃ (b : variable) (fields_b : unboxed_fields),
      alloc_field d f Value (Var b) ∧ fields_of M b = Some fields_b ∧ x <> b
      ∧ Forall2
          (fun (fu : field * unboxed_field)
               (vks : variable * flambda_kind * simple) =>
             let '(f, u) := fu in
             let '(v, kind, s) := vks in
             u = Not_unboxed v ∧ kind = field_kind f
             ∧ ∃ v' : variable, s = Var v' ∧ (f, Not_unboxed v') ∈ fields_b)
          nested l
  end.

(* The parameters of a handler as each run has them, of the shapes [sh]:
   one not unboxed ([None]) is a parameter of both; one unboxed with the
   fields [fields] ([Some fields]), of kind [Value], is in the second the
   variables of its fields. *)
Inductive params_rel : list (option unboxed_fields)
    -> list (variable * flambda_kind) -> list (variable * flambda_kind)
    -> Prop :=
  | params_rel_nil : params_rel [] [] []
  | params_rel_plain (sh : list (option unboxed_fields)) (p : variable)
      (kind : flambda_kind) (ps ps' : list (variable * flambda_kind)) :
    plain_variable M p = true -> params_rel sh ps ps' ->
    params_rel (None :: sh) ((p, kind) :: ps) ((p, kind) :: ps')
  | params_rel_unboxed (sh : list (option unboxed_fields)) (p : variable)
      (fields : unboxed_fields) (ps ps' : list (variable * flambda_kind)) :
    fields_of M p = Some fields -> among p (field_variables M) = false ->
    flat_fields fields = true -> params_rel sh ps ps' ->
    params_rel (Some fields :: sh) ((p, Value) :: ps)
      (fields_leaves fields ++ ps').

(* The arguments given to a handler of the shapes [sh]: a simple of
   [plain_simple] in both; for an unboxed parameter, an unboxed variable
   in the first and, in the second, the variables of its fields that the
   parameter has. Missing arguments are missing in both, and those beyond
   the parameters are alike. *)
Inductive args_rel : list (option unboxed_fields) -> list simple
    -> list simple -> Prop :=
  | args_rel_beyond (args : list simple) :
    forallb (plain_simple M) args = true -> args_rel [] args args
  | args_rel_missing (o : option unboxed_fields)
      (sh : list (option unboxed_fields)) :
    args_rel (o :: sh) [] []
  | args_rel_plain (sh : list (option unboxed_fields)) (a : simple)
      (args args' : list simple) :
    plain_simple M a = true -> args_rel sh args args' ->
    args_rel (None :: sh) (a :: args) (a :: args')
  | args_rel_unboxed (sh : list (option unboxed_fields))
      (fields fields_y : unboxed_fields) (y : variable) (vs : list variable)
      (args args' : list simple) :
    fields_of M y = Some fields_y ->
    Forall2
      (fun (fu : field * unboxed_field) (v : variable) =>
         (∃ v0 : variable, snd fu = Not_unboxed v0)
         ∧ (fst fu, Not_unboxed v) ∈ fields_y)
      fields vs ->
    args_rel sh args args' ->
    args_rel (Some fields :: sh) (Var y :: args) (map Var vs ++ args').

(* The shapes of the parameters of a continuation; one that is not bound
   is a way out, whose arguments are alike. *)
Definition shapes_at (K : finite_map (list (option unboxed_fields)))
    (k : continuation) : list (option unboxed_fields) :=
  match lookup k K with
  | Some sh => sh
  | None => []
  end.

Definition ac_rel (K : finite_map (list (option unboxed_fields)))
    (ac ac' : apply_cont) : Prop :=
  ac_continuation ac' = ac_continuation ac
  ∧ ac_trap_action ac' = ac_trap_action ac
  ∧ args_rel (shapes_at K (ac_continuation ac)) (ac_args ac) (ac_args ac').

(* [K] are the shapes of the parameters of the continuations in scope. *)
Inductive ub : finite_map (list (option unboxed_fields)) -> expr -> expr
    -> Prop :=
  | ub_let (K : finite_map (list (option unboxed_fields))) (x : variable)
      (kind : flambda_kind) (d : named) (body body' : expr) :
    plain_variable M x = true -> named_plain d = true -> ub K body body' ->
    ub K (Let x kind d body) (Let x kind d body')
  | ub_alloc (K : finite_map (list (option unboxed_fields))) (x : variable)
      (d : named) (fields : unboxed_fields)
      (ls : list (list (variable * flambda_kind * simple)))
      (body body' : expr) :
    fields_of M x = Some fields ->
    among x (field_variables M) = false ->
    Forall2 (alloc_entry x d) fields ls ->
    ub K body body' ->
    ub K (Let x Value d body) (lets (concat ls) body')
  | ub_alias (K : finite_map (list (option unboxed_fields))) (x y : variable)
      (fields fields_y : unboxed_fields)
      (l : list (variable * flambda_kind * simple)) (body body' : expr) :
    fields_of M x = Some fields ->
    among x (field_variables M) = false ->
    fields_of M y = Some fields_y -> x <> y ->
    Forall2
      (fun (fu : field * unboxed_field)
           (vks : variable * flambda_kind * simple) =>
         let '(f, u) := fu in
         let '(v, kind, s) := vks in
         u = Not_unboxed v ∧ kind = field_kind f
         ∧ ∃ v' : variable, s = Var v' ∧ (f, Not_unboxed v') ∈ fields_y)
      fields l ->
    ub K body body' ->
    ub K (Let x Value (Simple (Var y)) body) (lets l body')
  | ub_load (K : finite_map (list (option unboxed_fields))) (x : variable)
      (kind : flambda_kind) (d : named) (f : field) (s : simple)
      (y : variable) (fields fields_y nested : unboxed_fields)
      (l : list (variable * flambda_kind * simple)) (body body' : expr) :
    fields_of M x = Some fields ->
    among x (field_variables M) = false ->
    read_of d = Some (f, s) ->
    (s = Var y ∨ ∃ k : flambda_kind, s = Const (Poison k)) ->
    fields_of M y = Some fields_y -> x <> y ->
    (f, Unboxed nested) ∈ fields_y ->
    Forall2
      (fun (fu : field * unboxed_field)
           (vks : variable * flambda_kind * simple) =>
         let '(f, u) := fu in
         let '(v, kind, s) := vks in
         u = Not_unboxed v ∧ kind = field_kind f
         ∧ ∃ v' : variable, s = Var v' ∧ (f, Not_unboxed v') ∈ nested)
      fields l ->
    ub K body body' ->
    ub K (Let x Value d body) (lets l body')
  | ub_read (K : finite_map (list (option unboxed_fields))) (x : variable)
      (kind : flambda_kind) (d : named) (f : field)
      (s : simple) (y v : variable) (fields : unboxed_fields)
      (body body' : expr) :
    plain_variable M x = true ->
    read_of d = Some (f, s) ->
    (s = Var y ∨ ∃ k : flambda_kind, s = Const (Poison k)) ->
    fields_of M y = Some fields ->
    (f, Not_unboxed v) ∈ fields ->
    ub K body body' ->
    ub K (Let x kind d body) (Let x kind (Simple (Var v)) body')
  | ub_static (K : finite_map (list (option unboxed_fields)))
      (bindings : list static_binding) (body body' : expr) :
    forallb (local_binding M) bindings = true -> ub K body body' ->
    ub K (Let_static bindings body) (Let_static bindings body')
  | ub_let_cont (K : finite_map (list (option unboxed_fields)))
      (k : continuation) (sh : list (option unboxed_fields))
      (ps ps' : list (variable * flambda_kind))
      (metadata metadata' : handler_metadata)
      (handler handler' body body' : expr) :
    params_rel sh ps ps' -> hm_params_kinds metadata = map snd ps ->
    hm_params_kinds metadata' = map snd ps' -> ub K handler handler' ->
    ub ((k, sh) :: K) body body' ->
    ub K
      (Let_cont_non_recursive k (Handler (map fst ps) metadata handler) body)
      (Let_cont_non_recursive k (Handler (map fst ps') metadata' handler')
         body')
  | ub_let_cont_rec (K S : finite_map (list (option unboxed_fields)))
      (handlers handlers' : list (continuation * continuation_handler))
      (body body' : expr) :
    ub_handlers (S ++ K) handlers handlers' S ->
    handlers_bind_statics handlers = handlers_bind_statics handlers' ->
    ub (S ++ K) body body' ->
    ub K (Let_cont_recursive handlers body)
      (Let_cont_recursive handlers' body')
  | ub_apply_cont (K : finite_map (list (option unboxed_fields)))
      (ac ac' : apply_cont) :
    ac_rel K ac ac' -> ub K (Apply_cont ac) (Apply_cont ac')
  | ub_switch (K : finite_map (list (option unboxed_fields)))
      (scrutinee : simple) (arms arms' : list (Z * apply_cont)) :
    plain_simple M scrutinee = true ->
    Forall2
      (fun (arm arm' : Z * apply_cont) =>
         fst arm' = fst arm ∧ ac_rel K (snd arm) (snd arm'))
      arms arms' ->
    ub K (Switch scrutinee arms) (Switch scrutinee arms')
  | ub_invalid (K : finite_map (list (option unboxed_fields))) :
    ub K Invalid Invalid
with ub_handlers : finite_map (list (option unboxed_fields))
    -> list (continuation * continuation_handler)
    -> list (continuation * continuation_handler)
    -> finite_map (list (option unboxed_fields)) -> Prop :=
  | ub_handlers_nil (K : finite_map (list (option unboxed_fields))) :
    ub_handlers K [] [] []
  | ub_handlers_cons (K : finite_map (list (option unboxed_fields)))
      (k : continuation) (sh : list (option unboxed_fields))
      (ps ps' : list (variable * flambda_kind))
      (metadata metadata' : handler_metadata) (handler handler' : expr)
      (handlers handlers' : list (continuation * continuation_handler))
      (S : finite_map (list (option unboxed_fields))) :
    params_rel sh ps ps' -> hm_params_kinds metadata = map snd ps ->
    hm_params_kinds metadata' = map snd ps' -> ub K handler handler' ->
    ub_handlers K handlers handlers' S ->
    ub_handlers K ((k, Handler (map fst ps) metadata handler) :: handlers)
      ((k, Handler (map fst ps') metadata' handler') :: handlers')
      ((k, sh) :: S).

(* The environments: a variable of neither kind has one value in both;
   an unboxed variable bound in the first has each of its fields unboxed
   bound in the second to the field of its value, of the kind of the
   field. *)
Definition env_rel (vars vars' : env) : Prop :=
  (∀ z : variable, plain_variable M z = true -> lookup z vars = lookup z vars')
  ∧ (∀ (x : variable) (fields : unboxed_fields) (w : value),
       fields_of M x = Some fields -> lookup x vars = Some w ->
       ∀ (f : field) (v : variable),
         (f, Not_unboxed v) ∈ fields ->
         ∃ w' : value,
           field_value f w = Some w' ∧ lookup v vars' = Some w'
           ∧ kind_of_value w' = field_kind f)
  ∧ (∀ (x : variable) (fields : unboxed_fields) (w : value),
       fields_of M x = Some fields -> lookup x vars = Some w ->
       ∀ (f : field) (nested : unboxed_fields),
         (f, Unboxed nested) ∈ fields ->
         ∃ w1 : value,
           field_value f w = Some w1
           ∧ ∀ (f' : field) (v : variable),
               (f', Not_unboxed v) ∈ nested ->
               ∃ w' : value,
                 field_value f' w1 = Some w' ∧ lookup v vars' = Some w'
                 ∧ kind_of_value w' = field_kind f').

Inductive cont_rel : list (option unboxed_fields) -> cont_value -> cont_value
    -> Prop :=
  | cont_rel_closure (sh : list (option unboxed_fields))
      (ps ps' : list (variable * flambda_kind))
      (metadata metadata' : handler_metadata) (handler handler' : expr)
      (vars vars' : env) (K S : finite_map (list (option unboxed_fields)))
      (conts conts' : cont_env)
      (recursive recursive' : list (continuation * continuation_handler)) :
    params_rel sh ps ps' -> hm_params_kinds metadata = map snd ps ->
    hm_params_kinds metadata' = map snd ps' -> ub (S ++ K) handler handler' ->
    env_rel vars vars' -> conts_rel K conts conts' ->
    ub_handlers (S ++ K) recursive recursive' S ->
    cont_rel sh
      (Closure (Handler (map fst ps) metadata handler) vars conts recursive)
      (Closure (Handler (map fst ps') metadata' handler') vars' conts'
         recursive')
with conts_rel : finite_map (list (option unboxed_fields)) -> cont_env
    -> cont_env -> Prop :=
  | conts_rel_nil : conts_rel [] [] []
  | conts_rel_cons (k : continuation) (sh : list (option unboxed_fields))
      (c c' : cont_value) (K : finite_map (list (option unboxed_fields)))
      (conts conts' : cont_env) :
    cont_rel sh c c' -> conts_rel K conts conts' ->
    conts_rel ((k, sh) :: K) ((k, c) :: conts) ((k, c') :: conts').

Definition related (s t : state) : Prop :=
  ∃ K : finite_map (list (option unboxed_fields)),
    ub K (state_expr s) (state_expr t) ∧ env_rel (state_vars s) (state_vars t)
    ∧ conts_rel K (state_conts s) (state_conts t)
    ∧ state_statics s = state_statics t ∧ state_heap s = state_heap t.

End Related.

(* ======================================================================== *)
(* The variables of the fields                                              *)
(* ======================================================================== *)

Lemma node_eqb_decides : decides_equality node_eqb.
Proof.
  intros n n'. split; [ apply node_eqb_eq | intros <-; apply node_eqb_refl ].
Qed.

Section Fields.

Variable M : list (node * unboxed_fields).
Hypothesis M_distinct : distinct (field_variables M) = true.

Definition leaf_variables (fields : unboxed_fields) : list variable :=
  map fst (fields_leaves fields).

Lemma leaves_cons (f : field) (u : unboxed_field) (fields : unboxed_fields) :
  fields_leaves ((f, u) :: fields)
  = unboxed_field_leaves (field_kind f) u ++ fields_leaves fields.
Proof. reflexivity. Qed.

Lemma leaf_in (fields : unboxed_fields) (f : field) (v : variable) :
  (f, Not_unboxed v) ∈ fields -> v ∈ leaf_variables fields.
Proof.
  induction fields as [ | [f' u] fields IH]; intros Hin; [ destruct Hin | ].
  unfold leaf_variables. rewrite leaves_cons, map_app. apply in_or_app.
  destruct Hin as [Heq | Hin].
  - injection Heq as -> ->. left. left. reflexivity.
  - right. exact (IH Hin).
Qed.

Lemma fields_of_in (x : variable) (fields : unboxed_fields) :
  fields_of M x = Some fields -> (Node_variable x, fields) ∈ M.
Proof. apply (keyed_lookup_in node_eqb node_eqb_decides). Qed.

Lemma field_variables_in (x : variable) (fields : unboxed_fields)
    (v : variable) :
  fields_of M x = Some fields -> v ∈ leaf_variables fields ->
  v ∈ field_variables M.
Proof.
  intros Hx Hv. apply fields_of_in in Hx. unfold field_variables.
  apply in_flat_map. exists (Node_variable x, fields). split; assumption.
Qed.

Lemma plain_variable_iff (z : variable) :
  plain_variable M z = true
  <-> Keyed.is_bound node_eqb (Node_variable z) M = false
      ∧ among z (field_variables M) = false.
Proof.
  unfold plain_variable. rewrite andb_true_iff, !negb_true_iff. reflexivity.
Qed.

Lemma plain_not_unboxed (z : variable) (fields : unboxed_fields) :
  plain_variable M z = true -> fields_of M z = Some fields -> False.
Proof.
  intros Hz Hf. apply plain_variable_iff in Hz as [Hz _].
  unfold Keyed.is_bound in Hz. unfold fields_of in Hf. rewrite Hf in Hz.
  discriminate.
Qed.

Lemma plain_not_field (z x : variable) (fields : unboxed_fields)
    (f : field) :
  plain_variable M z = true -> fields_of M x = Some fields ->
  (f, Not_unboxed z) ∈ fields -> False.
Proof.
  intros Hz Hx Hin. apply plain_variable_iff in Hz as [_ Hz].
  pose proof (field_variables_in x fields z Hx (leaf_in fields f z Hin))
    as Hv.
  apply among_In in Hv. rewrite Hv in Hz. discriminate.
Qed.

(* A variable of a field of a field unboxed in turn is a variable of the
   fields. *)
Lemma nested_leaf_in (fields : unboxed_fields) (f : field)
    (nested : unboxed_fields) (f' : field) (v : variable) :
  (f, Unboxed nested) ∈ fields -> (f', Not_unboxed v) ∈ nested ->
  v ∈ leaf_variables fields.
Proof.
  induction fields as [ | [f0 u] fields IH]; intros Hin Hv; [ destruct Hin | ].
  unfold leaf_variables. rewrite leaves_cons, map_app. apply in_or_app.
  destruct Hin as [Heq | Hin].
  - injection Heq as -> ->. left. exact (leaf_in nested f' v Hv).
  - right. exact (IH Hin Hv).
Qed.

Lemma plain_not_nested_field (z x : variable) (fields : unboxed_fields)
    (f : field) (nested : unboxed_fields) (f' : field) :
  plain_variable M z = true -> fields_of M x = Some fields ->
  (f, Unboxed nested) ∈ fields -> (f', Not_unboxed z) ∈ nested -> False.
Proof.
  intros Hz Hx Hin Hv. apply plain_variable_iff in Hz as [_ Hz].
  pose proof (field_variables_in x fields z Hx
                (nested_leaf_in fields f nested f' z Hin Hv))
    as Hfv.
  apply among_In in Hfv. rewrite Hfv in Hz. discriminate.
Qed.

Lemma flat_no_nested (fields : unboxed_fields) (f : field)
    (nested : unboxed_fields) :
  flat_fields fields = true -> (f, Unboxed nested) ∈ fields -> False.
Proof.
  intros Hflat Hin. unfold flat_fields in Hflat.
  rewrite forallb_forall in Hflat.
  specialize (Hflat _ Hin). discriminate Hflat.
Qed.

Lemma nodup_app_apart {A : Type} (l l' : list A) (a : A) :
  NoDup (l ++ l') -> a ∈ l -> a ∈ l' -> False.
Proof.
  induction l as [ | b l IH]; intros Hnd Ha Ha'; [ destruct Ha | ].
  simpl in Hnd. inversion Hnd as [ | b' l0 Hb Hnd' [Heq Heq']]; subst.
  destruct Ha as [<- | Ha].
  - apply Hb. apply in_or_app. right. exact Ha'.
  - exact (IH Hnd' Ha Ha').
Qed.

(* The variables of two entries of a list without repetition are
   apart. *)
Lemma flat_map_nodup_apart {A : Type} (g : A -> list positive)
    (l : list A) (a b : A) :
  NoDup (flat_map g l) -> a ∈ l -> b ∈ l -> a <> b ->
  ∀ v : positive, v ∈ g a -> v ∈ g b -> False.
Proof.
  induction l as [ | c l IH]; intros Hnd Ha Hb Hne v Hva Hvb; [ destruct Ha | ].
  simpl in Hnd. apply NoDup_app_remove_l in Hnd as Hrest.
  destruct Ha as [<- | Ha]; destruct Hb as [<- | Hb].
  - exact (Hne eq_refl).
  - assert (Hv : v ∈ flat_map g l)
      by (apply in_flat_map; exists b; split; assumption).
    exact (nodup_app_apart _ _ v Hnd Hva Hv).
  - assert (Hv : v ∈ flat_map g l)
      by (apply in_flat_map; exists a; split; assumption).
    exact (nodup_app_apart _ _ v Hnd Hvb Hv).
  - exact (IH Hrest Ha Hb Hne v Hva Hvb).
Qed.

Lemma flat_map_nodup_one {A : Type} (g : A -> list positive) (l : list A)
    (a : A) :
  NoDup (flat_map g l) -> a ∈ l -> NoDup (g a).
Proof.
  induction l as [ | c l IH]; intros Hnd Ha; [ destruct Ha | ].
  simpl in Hnd. destruct Ha as [<- | Ha].
  - exact (NoDup_app_remove_r _ _ Hnd).
  - exact (IH (NoDup_app_remove_l _ _ Hnd) Ha).
Qed.

Lemma field_variables_nodup : NoDup (field_variables M).
Proof. apply distinct_NoDup. exact M_distinct. Qed.

(* The variables of the fields of one unboxed variable are distinct, and
   apart from those of another. *)
Lemma leaves_nodup (x : variable) (fields : unboxed_fields) :
  fields_of M x = Some fields -> NoDup (leaf_variables fields).
Proof.
  intros Hx. apply fields_of_in in Hx.
  exact (flat_map_nodup_one (fun '(_, fields) => leaf_variables fields) M
           _ field_variables_nodup Hx).
Qed.

Lemma leaves_apart (x x' : variable) (fields fields' : unboxed_fields)
    (v : variable) :
  fields_of M x = Some fields -> fields_of M x' = Some fields' -> x <> x' ->
  v ∈ leaf_variables fields -> v ∈ leaf_variables fields' -> False.
Proof.
  intros Hx Hx' Hne. apply fields_of_in in Hx, Hx'.
  apply (flat_map_nodup_apart (fun '(_, fields) => leaf_variables fields) M
           _ _ field_variables_nodup Hx Hx').
  intros [= Heq _]. exact (Hne Heq).
Qed.

End Fields.

(* ======================================================================== *)
(* Related environments                                                     *)
(* ======================================================================== *)

Section Envs.

Variable M : list (node * unboxed_fields).
Hypothesis M_distinct : distinct (field_variables M) = true.

Lemma plain_simple_eval (vars vars' : env) (s : simple) :
  env_rel M vars vars' -> plain_simple M s = true ->
  eval_simple vars s = eval_simple vars' s.
Proof.
  intros [Hplain _] Hs. destruct s as [x | sym | c]; simpl; try reflexivity.
  exact (Hplain x Hs).
Qed.

Lemma plain_simples_eval (vars vars' : env) (args : list simple) :
  env_rel M vars vars' -> forallb (plain_simple M) args = true ->
  eval_simples vars args = eval_simples vars' args.
Proof.
  intros Hrel. induction args as [ | s args IH]; simpl; [ reflexivity | ].
  intros Hargs. apply andb_true_iff in Hargs as [Hs Hargs].
  rewrite (plain_simple_eval vars vars' s Hrel Hs), (IH Hargs).
  reflexivity.
Qed.

Lemma named_plain_eval (statics : static_env) (vars vars' : env) (d : named) :
  env_rel M vars vars' -> named_plain M d = true ->
  eval_named statics vars d = eval_named statics vars' d.
Proof.
  intros Hrel Hd. destruct d as [s | p args | | | | | ]; simpl in Hd |- *;
    try discriminate Hd.
  - exact (plain_simple_eval vars vars' s Hrel Hd).
  - rewrite (plain_simples_eval vars vars' args Hrel Hd). reflexivity.
Qed.

Lemma local_bindings_eval (vars vars' : env) (bindings : list static_binding) :
  env_rel M vars vars' -> forallb (local_binding M) bindings = true ->
  ∀ statics : static_env,
    eval_static_bindings vars bindings statics
    = eval_static_bindings vars' bindings statics.
Proof.
  intros Hrel. induction bindings as [ | binding bindings IH]; simpl;
    [ reflexivity | ].
  intros Hb statics. apply andb_true_iff in Hb as [Hb Hbs].
  destruct binding as [id c | id | sym [ | | tag fields | | | ]];
    simpl in Hb; try discriminate Hb.
  destruct (lookup sym (static_values statics)); [ reflexivity | ].
  simpl. rewrite (plain_simples_eval vars vars' fields Hrel Hb).
  destruct (eval_simples vars' fields); simpl; [ apply IH, Hbs | reflexivity ].
Qed.

Lemma env_rel_bind (vars vars' : env) (x : variable) (w : value) :
  env_rel M vars vars' -> plain_variable M x = true ->
  env_rel M ((x, w) :: vars) ((x, w) :: vars').
Proof.
  intros [Hplain [Hfields Hnested]] Hx. split; [ | split ].
  - intros z Hz. rewrite !lookup_cons. destruct (Pos.eqb z x);
      [ reflexivity | exact (Hplain z Hz) ].
  - intros y fields w0 Hy Hlookup f v Hin.
    rewrite lookup_other in Hlookup
      by (intros ->; exact (plain_not_unboxed M x fields Hx Hy)).
    destruct (Hfields y fields w0 Hy Hlookup f v Hin) as (w' & Hw' & Hv & Hk).
    exists w'. split; [ exact Hw' | split; [ | exact Hk ] ].
    rewrite lookup_other; [ exact Hv | ].
    intros ->. exact (plain_not_field M x y fields f Hx Hy Hin).
  - intros y fields w0 Hy Hlookup f nested Hin.
    rewrite lookup_other in Hlookup
      by (intros ->; exact (plain_not_unboxed M x fields Hx Hy)).
    destruct (Hnested y fields w0 Hy Hlookup f nested Hin) as (w1 & Hw1 & Hn).
    exists w1. split; [ exact Hw1 | ]. intros f' v Hv.
    destruct (Hn f' v Hv) as (w' & Hw' & Hl & Hk).
    exists w'. split; [ exact Hw' | split; [ | exact Hk ] ].
    rewrite lookup_other; [ exact Hl | ]. intros ->.
    exact (plain_not_nested_field M x y fields f nested f' Hx Hy Hin Hv).
Qed.

Lemma env_rel_empty : env_rel M [] [].
Proof.
  split; [ reflexivity | split ]; intros x fields w _ Hlookup;
    discriminate Hlookup.
Qed.

Lemma leaf_variables_flat (f : field) (v : variable)
    (fields : unboxed_fields) :
  leaf_variables ((f, Not_unboxed v) :: fields) = v :: leaf_variables fields.
Proof. reflexivity. Qed.

(* The variables of the fields of [p], bound to the fields of [w]. *)
Lemma bind_leaves (w : value) (fields : unboxed_fields) (ws : list value) :
  Forall2
    (fun (fu : field * unboxed_field) (w' : value) =>
       (∃ v : variable, snd fu = Not_unboxed v)
       ∧ field_value (fst fu) w = Some w'
       ∧ kind_of_value w' = field_kind (fst fu))
    fields ws ->
  ∀ vars vars1 : env,
    NoDup (leaf_variables fields) ->
    bind_params (leaf_variables fields) ws vars = Some vars1 ->
    (∀ z : variable,
       z ∉ leaf_variables fields -> lookup z vars1 = lookup z vars)
    ∧ ∀ (f : field) (v : variable),
        (f, Not_unboxed v) ∈ fields ->
        ∃ w' : value,
          field_value f w = Some w' ∧ lookup v vars1 = Some w'
          ∧ kind_of_value w' = field_kind f.
Proof.
  induction 1 as [ | [f u] w1 fields ws Hfu _ IH]; intros vars vars1 Hnd
    Hbind.
  - simpl in Hbind. injection Hbind as <-. split; [ reflexivity | ].
    intros f v [].
  - destruct Hfu as ((v1 & Hu) & Hw1 & Hk1). simpl in Hu, Hw1, Hk1. subst u.
    rewrite leaf_variables_flat in Hnd, Hbind |- *.
    inversion Hnd as [ | v0 l0 Hv1 Hnd' [Heq0 Heq1]]; subst.
    simpl in Hbind.
    destruct (IH _ _ Hnd' Hbind) as [Hother Hfields].
    split.
    + intros z Hz. rewrite Hother by (intros Hin; apply Hz; right; exact Hin).
      rewrite lookup_other; [ reflexivity | ].
      intros ->. apply Hz. left. reflexivity.
    + intros f' v [Heq | Hin].
      * injection Heq as -> ->. exists w1. split; [ exact Hw1 | ].
        split; [ | exact Hk1 ]. rewrite (Hother v Hv1). apply lookup_head.
      * exact (Hfields f' v Hin).
Qed.

Lemma forall2_in_l {A B : Type} (P : A -> B -> Prop) (l : list A)
    (l' : list B) (a : A) :
  Forall2 P l l' -> a ∈ l -> ∃ b : B, P a b.
Proof.
  induction 1 as [ | a0 b0 l l' H _ IH]; intros Hin; [ destruct Hin | ].
  destruct Hin as [<- | Hin]; [ exists b0; exact H | exact (IH Hin) ].
Qed.

(* The parameter [p] unboxed bound to [w] in the first run, the
   variables of its fields to the fields of [w] in the second. *)
Lemma env_rel_bind_unboxed (vars vars' vars1 : env) (p : variable)
    (fields : unboxed_fields) (w : value) (ws : list value) :
  env_rel M vars vars' -> fields_of M p = Some fields ->
  Forall2
    (fun (fu : field * unboxed_field) (w' : value) =>
       (∃ v : variable, snd fu = Not_unboxed v)
       ∧ field_value (fst fu) w = Some w'
       ∧ kind_of_value w' = field_kind (fst fu))
    fields ws ->
  bind_params (leaf_variables fields) ws vars' = Some vars1 ->
  env_rel M ((p, w) :: vars) vars1.
Proof.
  intros [Hplain [Hfs Hnested]] Hp Hws Hbind.
  destruct (bind_leaves w fields ws Hws vars' vars1
              (leaves_nodup M M_distinct p fields Hp) Hbind)
    as [Hother Hleaves].
  split; [ | split ].
  - intros z Hz. rewrite lookup_other
      by (intros ->; exact (plain_not_unboxed M p fields Hz Hp)).
    rewrite (Hplain z Hz), Hother; [ reflexivity | ].
    intros Hin. apply plain_variable_iff in Hz as [_ Hz].
    pose proof (field_variables_in M p fields z Hp Hin) as Hfv.
    apply among_In in Hfv. rewrite Hfv in Hz. discriminate.
  - intros x fx w0 Hx Hlookup f v Hin.
    destruct (Pos.eq_dec x p) as [-> | Hne].
    + rewrite lookup_head in Hlookup. injection Hlookup as <-.
      rewrite Hp in Hx. injection Hx as <-. exact (Hleaves f v Hin).
    + rewrite lookup_other in Hlookup by exact Hne.
      destruct (Hfs x fx w0 Hx Hlookup f v Hin) as (w' & Hfw & Hv & Hk).
      exists w'. split; [ exact Hfw | split; [ | exact Hk ] ].
      rewrite Hother; [ exact Hv | ]. intros Hvp.
      exact (leaves_apart M M_distinct x p fx fields v Hx Hp Hne
               (leaf_in fx f v Hin) Hvp).
  - intros x fx w0 Hx Hlookup f nested Hin.
    destruct (Pos.eq_dec x p) as [-> | Hne].
    + exfalso. rewrite Hp in Hx. injection Hx as <-.
      destruct (forall2_in_l _ _ _ _ Hws Hin) as (w' & (v & Hv) & _).
      discriminate Hv.
    + rewrite lookup_other in Hlookup by exact Hne.
      destruct (Hnested x fx w0 Hx Hlookup f nested Hin) as (w1 & Hw1 & Hn).
      exists w1. split; [ exact Hw1 | ]. intros f' v Hv.
      destruct (Hn f' v Hv) as (w' & Hfw & Hl & Hk).
      exists w'. split; [ exact Hfw | split; [ | exact Hk ] ].
      rewrite Hother; [ exact Hl | ]. intros Hvp.
      exact (leaves_apart M M_distinct x p fx fields v Hx Hp Hne
               (nested_leaf_in fx f nested f' v Hin Hv) Hvp).
Qed.

Lemma resolve_rel (K : finite_map (list (option unboxed_fields)))
    (conts conts' : cont_env) (k : continuation) :
  conts_rel M K conts conts' ->
  (lookup k K = None ∧ resolve k conts = Outer k
   ∧ resolve k conts' = Outer k)
  ∨ ∃ sh : list (option unboxed_fields),
      lookup k K = Some sh ∧ cont_rel M sh (resolve k conts) (resolve k conts').
Proof.
  intros Hrel. unfold resolve.
  induction Hrel as [ | k' sh c c' K conts conts' Hc _ IH]; simpl;
    [ left; split; [ reflexivity | split; reflexivity ] | ].
  rewrite ?lookup_cons. destruct (Pos.eqb k k').
  - right. exists sh. split; [ reflexivity | exact Hc ].
  - exact IH.
Qed.

Lemma rec_closures_rel_from
    (KK K S S0 : finite_map (list (option unboxed_fields)))
    (all all' hs hs' : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) :
  KK = S ++ K ->
  ub_handlers M KK all all' S -> ub_handlers M KK hs hs' S0 ->
  env_rel M vars vars' -> conts_rel M K conts conts' ->
  conts_rel M (S0 ++ K)
    (map (fun '(k, h) => (k, Closure h vars conts all)) hs ++ conts)
    (map (fun '(k, h) => (k, Closure h vars' conts' all')) hs' ++ conts').
Proof.
  intros HKK Hall Hhs Hvars Hconts.
  induction Hhs as [ | KK0 k sh ps ps' md md' h h' hs hs' S1 Hps Hmd Hmd' Hh
                       _ IH];
    simpl; [ exact Hconts | ].
  constructor; [ | exact (IH HKK Hall) ].
  subst KK0. econstructor; eassumption.
Qed.

Lemma rec_closures_rel (K S : finite_map (list (option unboxed_fields)))
    (handlers handlers' : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) :
  ub_handlers M (S ++ K) handlers handlers' S -> env_rel M vars vars' ->
  conts_rel M K conts conts' ->
  conts_rel M (S ++ K) (rec_closures handlers vars conts ++ conts)
    (rec_closures handlers' vars' conts' ++ conts').
Proof.
  intros Hh.
  exact (rec_closures_rel_from (S ++ K) K S S _ _ _ _ vars vars' conts conts'
           eq_refl Hh Hh).
Qed.

End Envs.

(* ======================================================================== *)
(* The shapes of the continuations that are not free                        *)
(* ======================================================================== *)

Scheme ub_mut := Induction for ub Sort Prop
  with ub_handlers_mut := Induction for ub_handlers Sort Prop.

Section Agree.

Variable M : list (node * unboxed_fields).

Definition agree_on (l : list continuation)
    (K K' : finite_map (list (option unboxed_fields))) : Prop :=
  ∀ k : continuation, k ∈ l -> lookup k K' = lookup k K.

Lemma in_without_iff (a : positive) (bound names : list positive) :
  a ∈ without bound names <-> a ∈ names ∧ a ∉ bound.
Proof.
  unfold without. rewrite filter_In, negb_true_iff.
  split; intros [Hin Hbound]; (split; [ exact Hin | ]).
  - intros Ha. apply among_In in Ha. congruence.
  - destruct (among a bound) eqn:Hamong; [ | reflexivity ].
    apply among_In in Hamong. contradiction.
Qed.

Lemma ub_handlers_keys (K : finite_map (list (option unboxed_fields)))
    (hs hs' : list (continuation * continuation_handler))
    (S : finite_map (list (option unboxed_fields))) :
  ub_handlers M K hs hs' S -> map fst S = map fst hs.
Proof.
  induction 1 as [ | K k sh ps ps' md md' h h' hs hs' S _ _ _ _ _ IH];
    simpl; [ reflexivity | rewrite IH; reflexivity ].
Qed.

Lemma lookup_keys {A : Type} (k : positive) (l : list (positive * A)) :
  k ∈ map fst l -> lookup k l <> None.
Proof.
  induction l as [ | [k' a] l IH]; intros Hin; [ destruct Hin | ].
  rewrite lookup_cons. destruct (Pos.eqb k k') eqn:E; [ discriminate | ].
  apply IH. destruct Hin as [Heq | Hin]; [ | exact Hin ].
  simpl in Heq. subst k'. rewrite Pos.eqb_refl in E. discriminate.
Qed.

Lemma agree_app (S K K' : finite_map (list (option unboxed_fields)))
    (bound l : list continuation) :
  map fst S = bound -> agree_on (without bound l) K K' ->
  agree_on l (S ++ K) (S ++ K').
Proof.
  intros HS Hagree k Hk. rewrite !lookup_app.
  destruct (lookup k S) eqn:E; [ reflexivity | ].
  apply Hagree, in_without_iff. split; [ exact Hk | ].
  intros Hin. rewrite <- HS in Hin. exact (lookup_keys k S Hin E).
Qed.

(* The shapes of the continuations not free in a term do not matter. *)
Lemma ub_agree :
  ∀ (K : finite_map (list (option unboxed_fields))) (p q : expr),
    ub M K p q ->
    ∀ K' : finite_map (list (option unboxed_fields)),
      agree_on (free_continuations_of_expr p) K K' -> ub M K' p q.
Proof.
  apply (ub_mut M
           (fun K p q _ =>
              ∀ K', agree_on (free_continuations_of_expr p) K K' ->
                    ub M K' p q)
           (fun K hs hs' S _ =>
              ∀ K',
                agree_on
                  (flat_map (fun '(_, h) => free_continuations_of_handler h)
                     hs)
                  K K' ->
                ub_handlers M K' hs hs' S)).
  - intros K x kind d body body' Hx Hd _ IH K' Hagree.
    apply ub_let; [ exact Hx | exact Hd | apply IH, Hagree ].
  - intros K x d fields l body body' Hf Hxf Hl _ IH K' Hagree.
    apply ub_alloc with (fields := fields); try assumption.
    apply IH, Hagree.
  - intros K x y fields fields_y l body body' Hf Hxf Hy Hne Hl _ IH K' Hagree.
    apply (ub_alias M K' x y fields fields_y l); try assumption.
    apply IH, Hagree.
  - intros K x kind d f s y fields fields_y nested l body body' Hf Hxf Hr Hs
      Hy Hne Hin Hl _ IH K' Hagree.
    eapply ub_load; try eassumption. apply IH, Hagree.
  - intros K x kind d f s y v fields body body' Hx Hr Hs Hf Hin _ IH K'
      Hagree.
    eapply ub_read; try eassumption. apply IH, Hagree.
  - intros K bindings body body' Hb _ IH K' Hagree.
    apply ub_static; [ exact Hb | ]. apply IH.
    intros k Hk. apply Hagree. simpl. apply in_or_app. right. exact Hk.
  - intros K k sh ps ps' md md' h h' body body' Hps Hmd Hmd' _ IHh _ IHb K'
      Hagree.
    apply (ub_let_cont M K' k sh ps ps' md md' h h' body body'); try assumption.
    + apply IHh. intros k' Hk. apply Hagree. simpl. apply in_or_app.
      left. exact Hk.
    + apply IHb. intros k' Hk. rewrite !lookup_cons.
      destruct (Pos.eqb k' k) eqn:E; [ reflexivity | ].
      apply Hagree. simpl. apply in_or_app. right. apply in_without_iff.
      split; [ exact Hk | ]. intros [Heq | []]. subst k'.
      rewrite Pos.eqb_refl in E. discriminate.
  - intros K S hs hs' body body' Hhs IHhs Hst _ IHb K' Hagree.
    pose proof (ub_handlers_keys _ _ _ _ Hhs) as Hkeys.
    apply (ub_let_cont_rec M K' S); [ apply IHhs | exact Hst | apply IHb ];
      apply (agree_app S K K' (map fst hs)); try exact Hkeys;
      intros k Hk; apply Hagree; simpl; apply in_without_iff;
      apply in_without_iff in Hk as [Hk Hnot]; (split; [ | exact Hnot ]);
      apply in_or_app; [ left | right ]; exact Hk.
  - intros K ac ac' (Hk & Htrap & Hargs) K' Hagree.
    apply ub_apply_cont. split; [ exact Hk | split; [ exact Htrap | ] ].
    unfold shapes_at. rewrite (Hagree (ac_continuation ac) (or_introl eq_refl)).
    exact Hargs.
  - intros K scrutinee arms arms' Hs Harms K' Hagree.
    apply ub_switch; [ exact Hs | ]. simpl in Hagree. clear Hs.
    induction Harms as [ | [d ac] [d' ac'] arms arms' [Hd (Hk & Htrap & Hargs)]
                         _ IH];
      constructor.
    + split; [ exact Hd | ]. split; [ exact Hk | split; [ exact Htrap | ] ].
      simpl in Hk, Htrap, Hargs |- *. unfold shapes_at in Hargs |- *.
      rewrite (Hagree (ac_continuation ac) (or_introl eq_refl)). exact Hargs.
    + apply IH. intros k Hin. apply Hagree. simpl. right. apply in_or_app.
      right. exact Hin.
  - intros K K' _. apply ub_invalid.
  - intros K K' _. apply ub_handlers_nil.
  - intros K k sh ps ps' md md' h h' hs hs' S Hps Hmd Hmd' _ IHh _ IHhs K'
      Hagree.
    apply ub_handlers_cons; try assumption.
    + apply IHh. intros k' Hk. apply Hagree. simpl. apply in_or_app. left.
      exact Hk.
    + apply IHhs. intros k' Hk. apply Hagree. simpl. apply in_or_app.
      right. exact Hk.
Qed.

End Agree.

(* ======================================================================== *)
(* What the unboxing makes                                                  *)
(* ======================================================================== *)

Definition lets_variable (vks : variable * flambda_kind * simple) : variable :=
  let '(v, _, _) := vks in v.

(* The [Let]s run, each binding its variable to the value of its simple,
   which reads none of the variables bound. *)
Lemma lets_run (l : list (variable * flambda_kind * simple)) :
  ∀ (body : expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap),
    NoDup (map lets_variable l) ->
    (∀ (v : variable) (kind : flambda_kind) (s : simple) (v' : variable),
       (v, kind, s) ∈ l -> v' ∈ map lets_variable l -> s <> Var v') ->
    (∀ (v : variable) (kind : flambda_kind) (s : simple),
       (v, kind, s) ∈ l ->
       ∃ w : value, eval_simple vars s = Some w ∧ kind_of_value w = kind) ->
    ∃ vars1 : env,
      run (length l) (State (lets l body) vars conts statics hp)
      = Step (State body vars1 conts statics hp)
      ∧ (∀ z : variable,
           z ∉ map lets_variable l -> lookup z vars1 = lookup z vars)
      ∧ (∀ (v : variable) (kind : flambda_kind) (s : simple) (w : value),
           (v, kind, s) ∈ l -> eval_simple vars s = Some w ->
           lookup v vars1 = Some w).
Proof.
  induction l as [ | [[v kind] s] l IH]; intros body vars conts statics hp
    Hnd Hread Heval.
  - exists vars. split; [ reflexivity | ]. split; [ reflexivity | ].
    intros v kind s w [].
  - simpl in Hnd. inversion Hnd as [ | v0 l0 Hv Hnd' [Heq0 Heq1]]; subst.
    destruct (IH (Let v kind (Simple s) body) vars conts statics hp Hnd')
      as (vars1 & Hrun & Hother & Hbound).
    { intros v1 kind1 s1 v' Hin Hv'. apply (Hread v1 kind1 s1 v');
        [ right; exact Hin | right; exact Hv' ]. }
    { intros v1 kind1 s1 Hin. apply (Heval v1 kind1 s1). right. exact Hin. }
    destruct (Heval v kind s (or_introl eq_refl)) as (w & Hw & Hkind).
    assert (Hs : eval_simple vars1 s = Some w).
    { rewrite <- Hw. destruct s as [z | sym | c]; simpl; try reflexivity.
      apply Hother. intros Hz.
      exact (Hread v kind (Var z) z (or_introl eq_refl) (or_intror Hz)
               eq_refl). }
    exists ((v, w) :: vars1). split; [ | split ].
    + change (lets ((v, kind, s) :: l) body)
        with (lets l (Let v kind (Simple s) body)).
      simpl length. rewrite <- Nat.add_1_r, run_add, Hrun, run_succ, step_let.
      cbn [eval_named]. rewrite Hs, Hkind, kind_eqb_refl. reflexivity.
    + intros z Hz. rewrite lookup_other by (intros ->; apply Hz; left;
        reflexivity).
      apply Hother. intros Hin. apply Hz. right. exact Hin.
    + intros v' kind' s' w' [Heq | Hin] Hw'.
      * injection Heq as -> -> ->. rewrite Hw in Hw'. injection Hw' as <-.
        apply lookup_head.
      * rewrite lookup_other; [ exact (Hbound v' kind' s' w' Hin Hw') | ].
        intros ->. apply Hv. apply in_map_iff. exists (v, kind', s').
        split; [ reflexivity | exact Hin ].
Qed.

Lemma is_number_of_kind_kind (bn : boxable_number) (c : const) :
  is_number_of_kind bn c = true -> kind_of_const c = unboxed_kind bn.
Proof.
  destruct bn, c; simpl; intros H; try discriminate H; reflexivity.
Qed.

(* The variable of a field of an allocation that is made is bound to the
   field. *)
Lemma alloc_field_eval (statics : static_env) (vars : env) (d : named)
    (w : value) (f : field) (kind : flambda_kind) (s : simple) :
  eval_named statics vars d = Some w -> alloc_field d f kind s ->
  ∃ w' : value,
    eval_simple vars s = Some w' ∧ field_value f w = Some w'
    ∧ kind_of_value w' = kind.
Proof.
  intros Hd Hf. destruct d as [ | p args | | | | | ]; try contradiction.
  destruct p as [p | p | p | p]; try contradiction.
  - destruct p; try contradiction.
    destruct args as [ | c [ | ]]; try contradiction.
    destruct Hf as (-> & -> & ->). simpl in Hd.
    destruct (eval_simple vars c) as [v | ] eqn:Hc; [ | discriminate Hd ].
    simpl in Hd. destruct v as [c0 | | | | | | | | ]; try discriminate Hd.
    rewrite eval_prim_unary in Hd. cbn [eval_unary_primitive] in Hd.
    destruct (is_number_of_kind _ c0) eqn:Hk; [ | discriminate Hd ].
    injection Hd as <-. exists (Value_const c0).
    split; [ reflexivity | split; [ reflexivity | ] ].
    exact (is_number_of_kind_kind _ c0 Hk).
  - destruct p as [[tag] | ]; try contradiction.
    destruct Hf as (i & -> & -> & Hnth). simpl in Hd.
    destruct (eval_simples vars args) as [vs | ] eqn:Hargs;
      [ | discriminate Hd ].
    simpl in Hd. rewrite eval_prim_variadic in Hd.
    cbn [eval_variadic_primitive] in Hd.
    destruct (forallb (fun v => kind_eqb (kind_of_value v) Value) vs
              && forallb (bound_if_symbol statics) vs) eqn:Hok;
      [ | discriminate Hd ].
    injection Hd as <-. apply andb_true_iff in Hok as [Hkinds _].
    destruct (nth_error vs i) as [w' | ] eqn:Hw'.
    + exists w'. split; [ | split; [ exact Hw' | ] ].
      * destruct (FlambdaReaperFlowProofs.eval_simples_nth vars args vs i w'
                    Hargs Hw')
          as (s' & Hs' & Heval). rewrite Hnth in Hs'. injection Hs' as <-.
        exact Heval.
      * rewrite forallb_forall in Hkinds.
        apply kind_eqb_eq, Hkinds. exact (nth_error_In _ _ Hw').
    + exfalso. apply nth_error_None in Hw'.
      apply FlambdaReaperFlowProofs.eval_simples_length in Hargs.
      assert (Hlt : i < length args)
        by (apply nth_error_Some; rewrite Hnth; discriminate).
      lia.
Qed.

(* A read of a field: of the field of the value read, which is not a
   constant. *)
Lemma read_eval (statics : static_env) (vars : env) (d : named) (f : field)
    (s : simple) (r : value) :
  read_of d = Some (f, s) -> eval_named statics vars d = Some r ->
  ∃ w0 : value,
    eval_simple vars s = Some w0 ∧ (∀ c : const, w0 <> Value_const c)
    ∧ ∀ w' : value, field_value f w0 = Some w' -> r = w'.
Proof.
  intros Hread Hd.
  destruct d as [ | p args | | | | | ]; try discriminate Hread.
  destruct p as [p | p | p | p]; try discriminate Hread.
  destruct args as [ | a [ | ]]; destruct p; try discriminate Hread;
    simpl in Hread; injection Hread as <- <-; simpl in Hd;
    destruct (eval_simple vars a) as [w0 | ]; try discriminate Hd;
    simpl in Hd; rewrite eval_prim_unary in Hd;
    cbn [eval_unary_primitive] in Hd; exists w0; (split; [ reflexivity | ]).
  - (* Block_load *)
    destruct kind as [tag size field_kind]. unfold eval_block_load in Hd.
    destruct (block_of statics w0) as [[tag0 fields] | ] eqn:Hblock;
      [ | discriminate Hd ].
    simpl in Hd. split.
    + intros c ->. discriminate Hblock.
    + intros w' Hw'.
      destruct w0 as [ | | | | tagw fieldsw | | | | ]; try discriminate Hw'.
      simpl in Hw'. simpl in Hblock. injection Hblock as Htag Hfields.
      subst tagw fieldsw.
      destruct (known_is tag tag0
                && known_is size (Z.of_nat (length fields))
                && (0 <=? field)%Z); [ | discriminate Hd ].
      rewrite Hw' in Hd.
      destruct (match field_kind with
                | Any_value => kind_eqb (kind_of_value w') Value
                | Immediate => is_tagged_immediate w'
                end); [ | discriminate Hd ].
      injection Hd as <-. reflexivity.
  - (* Unbox_number *)
    destruct (boxed_number_of statics w0) as [c | ] eqn:Hbox;
      [ | discriminate Hd ].
    simpl in Hd. split.
    + intros c' ->. discriminate Hbox.
    + intros w' Hw'.
      destruct w0 as [ | | | | | | c0 | | ]; try discriminate Hw'.
      simpl in Hw'. injection Hw' as <-. simpl in Hbox.
      unfold boxable_contents in Hbox.
      destruct (boxable_number_of_const c0); [ | discriminate Hbox ].
      injection Hbox as <-.
      destruct (is_number_of_kind _ c0); [ | discriminate Hd ].
      injection Hd as <-. reflexivity.
Qed.

(* ======================================================================== *)
(* The simulation                                                           *)
(* ======================================================================== *)

Section Simulation.

Variable M : list (node * unboxed_fields).
Hypothesis M_distinct : distinct (field_variables M) = true.

Definition alloc_entries (d : named) : unboxed_fields
    -> list (variable * flambda_kind * simple) -> Prop :=
  Forall2
    (fun (fu : field * unboxed_field) (vks : variable * flambda_kind * simple)
     =>
       let '(f, u) := fu in
       let '(v, kind, s) := vks in
       u = Not_unboxed v ∧ alloc_field d f kind s ∧ plain_simple M s = true).

Lemma alloc_entries_l (d : named) (fields : unboxed_fields)
    (l : list (variable * flambda_kind * simple)) (v : variable)
    (kind : flambda_kind) (s : simple) :
  alloc_entries d fields l -> (v, kind, s) ∈ l ->
  ∃ f : field,
    (f, Not_unboxed v) ∈ fields ∧ alloc_field d f kind s
    ∧ plain_simple M s = true.
Proof.
  induction 1 as [ | [f u] [[v' kind'] s'] fields l Hfu _ IH]; intros Hin;
    [ destruct Hin | ].
  destruct Hin as [Heq | Hin].
  - injection Heq as -> -> ->. destruct Hfu as (-> & Hf & Hs).
    exists f. split; [ left; reflexivity | split; assumption ].
  - destruct (IH Hin) as (f' & Hin' & Hf & Hs).
    exists f'. split; [ right; exact Hin' | split; assumption ].
Qed.

Lemma alloc_entries_r (d : named) (fields : unboxed_fields)
    (l : list (variable * flambda_kind * simple)) (f : field)
    (v : variable) :
  alloc_entries d fields l -> (f, Not_unboxed v) ∈ fields ->
  ∃ (kind : flambda_kind) (s : simple),
    (v, kind, s) ∈ l ∧ alloc_field d f kind s ∧ plain_simple M s = true.
Proof.
  induction 1 as [ | [f' u] [[v' kind'] s'] fields l Hfu _ IH]; intros Hin;
    [ destruct Hin | ].
  destruct Hin as [Heq | Hin].
  - injection Heq as -> ->. destruct Hfu as ([= ->] & Hf & Hs).
    exists kind', s'. split; [ left; reflexivity | split; assumption ].
  - destruct (IH Hin) as (kind & s & Hin' & Hf & Hs).
    exists kind, s. split; [ right; exact Hin' | split; assumption ].
Qed.

Lemma alloc_entries_variables (d : named) (fields : unboxed_fields)
    (l : list (variable * flambda_kind * simple)) :
  alloc_entries d fields l -> map lets_variable l = leaf_variables fields.
Proof.
  induction 1 as [ | [f u] [[v kind] s] fields l Hfu _ IH]; [ reflexivity | ].
  destruct Hfu as (-> & _ & _). unfold leaf_variables in *.
  rewrite leaves_cons, map_app. simpl. rewrite IH. reflexivity.
Qed.

Definition alias_entries (fields_y : unboxed_fields) : unboxed_fields
    -> list (variable * flambda_kind * simple) -> Prop :=
  Forall2
    (fun (fu : field * unboxed_field) (vks : variable * flambda_kind * simple)
     =>
       let '(f, u) := fu in
       let '(v, kind, s) := vks in
       u = Not_unboxed v ∧ kind = field_kind f
       ∧ ∃ v' : variable, s = Var v' ∧ (f, Not_unboxed v') ∈ fields_y).

Lemma alias_entries_l (fields_y fields : unboxed_fields)
    (l : list (variable * flambda_kind * simple)) (v : variable)
    (kind : flambda_kind) (s : simple) :
  alias_entries fields_y fields l -> (v, kind, s) ∈ l ->
  ∃ f : field,
    (f, Not_unboxed v) ∈ fields ∧ kind = field_kind f
    ∧ ∃ v' : variable, s = Var v' ∧ (f, Not_unboxed v') ∈ fields_y.
Proof.
  induction 1 as [ | [f u] [[v' kind'] s'] fields l Hfu _ IH]; intros Hin;
    [ destruct Hin | ].
  destruct Hin as [Heq | Hin].
  - injection Heq as -> -> ->. destruct Hfu as (-> & Hk & Hs).
    exists f. split; [ left; reflexivity | split; assumption ].
  - destruct (IH Hin) as (f' & Hin' & Hk & Hs).
    exists f'. split; [ right; exact Hin' | split; assumption ].
Qed.

Lemma alias_entries_r (fields_y fields : unboxed_fields)
    (l : list (variable * flambda_kind * simple)) (f : field)
    (v : variable) :
  alias_entries fields_y fields l -> (f, Not_unboxed v) ∈ fields ->
  ∃ (kind : flambda_kind) (s : simple),
    (v, kind, s) ∈ l ∧ kind = field_kind f
    ∧ ∃ v' : variable, s = Var v' ∧ (f, Not_unboxed v') ∈ fields_y.
Proof.
  induction 1 as [ | [f' u] [[v' kind'] s'] fields l Hfu _ IH]; intros Hin;
    [ destruct Hin | ].
  destruct Hin as [Heq | Hin].
  - injection Heq as -> ->. destruct Hfu as ([= ->] & Hk & Hs).
    exists kind', s'. split; [ left; reflexivity | split; assumption ].
  - destruct (IH Hin) as (kind & s & Hin' & Hk & Hs).
    exists kind, s. split; [ right; exact Hin' | split; assumption ].
Qed.

Lemma alias_entries_variables (fields_y fields : unboxed_fields)
    (l : list (variable * flambda_kind * simple)) :
  alias_entries fields_y fields l ->
  map lets_variable l = leaf_variables fields.
Proof.
  induction 1 as [ | [f u] [[v kind] s] fields l Hfu _ IH]; [ reflexivity | ].
  destruct Hfu as (-> & _ & _). unfold leaf_variables in *.
  rewrite leaves_cons, map_app. simpl. rewrite IH. reflexivity.
Qed.

Lemma alias_entries_l' (fields_y fields : unboxed_fields)
    (l : list (variable * flambda_kind * simple)) (f : field)
    (nested : unboxed_fields) :
  alias_entries fields_y fields l -> (f, Unboxed nested) ∈ fields -> False.
Proof.
  intros Hl Hin. destruct (forall2_in_l _ _ _ _ Hl Hin) as ([[v k] s] & Hu & _).
  discriminate Hu.
Qed.

Lemma alloc_entry_flat (x : variable) (d : named) (fields : unboxed_fields)
    (l : list (variable * flambda_kind * simple)) :
  alloc_entries d fields l ->
  Forall2 (alloc_entry M x d) fields (map (fun e => [e]) l)
  ∧ concat (map (fun e => [e]) l) = l.
Proof.
  induction 1 as [ | [f u] [[v kind] s] fields l (-> & Hf & Hs) _ [IH IHc]];
    [ split; [ constructor | reflexivity ] | ].
  split; [ | cbn [map concat app]; rewrite IHc; reflexivity ].
  constructor; [ | exact IH ]. exists kind, s.
  split; [ reflexivity | split; assumption ].
Qed.

Lemma alloc_entries2_l (x : variable) (d : named) (fields : unboxed_fields)
    (ls : list (list (variable * flambda_kind * simple))) (v : variable)
    (kind : flambda_kind) (s : simple) :
  Forall2 (alloc_entry M x d) fields ls -> (v, kind, s) ∈ concat ls ->
  (∃ f : field,
     (f, Not_unboxed v) ∈ fields ∧ alloc_field d f kind s
     ∧ plain_simple M s = true)
  ∨ ∃ (f : field) (nested : unboxed_fields) (b : variable)
      (fields_b : unboxed_fields) (f' : field),
      (f, Unboxed nested) ∈ fields ∧ alloc_field d f Value (Var b)
      ∧ fields_of M b = Some fields_b ∧ x <> b
      ∧ (f', Not_unboxed v) ∈ nested ∧ kind = field_kind f'
      ∧ ∃ vb : variable, s = Var vb ∧ (f', Not_unboxed vb) ∈ fields_b.
Proof.
  induction 1 as [ | [f u] l fields ls Hfu _ IH]; intros Hin;
    [ destruct Hin | ].
  cbn [concat] in Hin. apply in_app_or in Hin as [Hin | Hin].
  - destruct u as [v0 | nested]; cbn [alloc_entry] in Hfu.
    + destruct Hfu as (kind0 & s0 & -> & Hf & Hs).
      destruct Hin as [[= -> -> ->] | []].
      left. exists f. split; [ left; reflexivity | split; assumption ].
    + destruct Hfu as (b & fb & Hf & Hb & Hxb & Hl).
      destruct (alias_entries_l fb nested l v kind s Hl Hin)
        as (f' & Hin' & Hk & vb & Hs & Hinb).
      right. exists f, nested, b, fb, f'.
      split; [ left; reflexivity | ].
      do 5 (split; [ assumption | ]). exists vb. split; assumption.
  - destruct (IH Hin) as [(f' & Hin' & Hrest) | (f' & n & b & fb & f'' & Hin'
                                                  & Hrest)].
    + left. exists f'. split; [ right; exact Hin' | exact Hrest ].
    + right. exists f', n, b, fb, f''. split; [ right; exact Hin' | ].
      exact Hrest.
Qed.

Lemma alloc_entries2_flat (x : variable) (d : named) (fields : unboxed_fields)
    (ls : list (list (variable * flambda_kind * simple))) (f : field)
    (v : variable) :
  Forall2 (alloc_entry M x d) fields ls -> (f, Not_unboxed v) ∈ fields ->
  ∃ (kind : flambda_kind) (s : simple),
    (v, kind, s) ∈ concat ls ∧ alloc_field d f kind s
    ∧ plain_simple M s = true.
Proof.
  induction 1 as [ | [f0 u] l fields ls Hfu _ IH]; intros Hin;
    [ destruct Hin | ].
  cbn [concat]. destruct Hin as [Heq | Hin].
  - injection Heq as -> ->. cbn [alloc_entry] in Hfu.
    destruct Hfu as (kind & s & -> & Hf & Hs).
    exists kind, s. split; [ left; reflexivity | split; assumption ].
  - destruct (IH Hin) as (kind & s & Hin' & Hrest).
    exists kind, s. split; [ apply in_or_app; right; exact Hin' | ].
    exact Hrest.
Qed.

Lemma alloc_entries2_nested (x : variable) (d : named)
    (fields : unboxed_fields)
    (ls : list (list (variable * flambda_kind * simple))) (f : field)
    (nested : unboxed_fields) :
  Forall2 (alloc_entry M x d) fields ls -> (f, Unboxed nested) ∈ fields ->
  ∃ (b : variable) (fields_b : unboxed_fields),
    alloc_field d f Value (Var b) ∧ fields_of M b = Some fields_b ∧ x <> b
    ∧ ∀ (f' : field) (v : variable),
        (f', Not_unboxed v) ∈ nested ->
        ∃ vb : variable,
          (v, field_kind f', Var vb) ∈ concat ls
          ∧ (f', Not_unboxed vb) ∈ fields_b.
Proof.
  induction 1 as [ | [f0 u] l fields ls Hfu _ IH]; intros Hin;
    [ destruct Hin | ].
  cbn [concat]. destruct Hin as [Heq | Hin].
  - injection Heq as -> ->. cbn [alloc_entry] in Hfu.
    destruct Hfu as (b & fb & Hf & Hb & Hxb & Hl).
    exists b, fb. do 3 (split; [ assumption | ]).
    intros f' v Hv.
    destruct (alias_entries_r fb nested l f' v Hl Hv)
      as (kind & s & Hin & -> & vb & -> & Hinb).
    exists vb. split; [ apply in_or_app; left; exact Hin | exact Hinb ].
  - destruct (IH Hin) as (b & fb & Hf & Hb & Hxb & Hall).
    exists b, fb. do 3 (split; [ assumption | ]).
    intros f' v Hv. destruct (Hall f' v Hv) as (vb & Hin' & Hinb).
    exists vb. split; [ apply in_or_app; right; exact Hin' | exact Hinb ].
Qed.

Lemma alloc_entries2_variables (x : variable) (d : named)
    (fields : unboxed_fields)
    (ls : list (list (variable * flambda_kind * simple))) :
  Forall2 (alloc_entry M x d) fields ls ->
  map lets_variable (concat ls) = leaf_variables fields.
Proof.
  induction 1 as [ | [f u] l fields ls Hfu _ IH]; [ reflexivity | ].
  cbn [concat]. rewrite map_app. unfold leaf_variables in *.
  rewrite leaves_cons, map_app, IH. f_equal.
  destruct u as [v | nested]; cbn [alloc_entry] in Hfu.
  - destruct Hfu as (kind & s & -> & _). reflexivity.
  - destruct Hfu as (b & fb & _ & _ & _ & Hl).
    exact (alias_entries_variables fb nested l Hl).
Qed.

Lemma run_answer (R : state -> state -> Prop) (s s' t t' : state) (n : nat) :
  run n t = Step t' -> R s' t' ->
  (n = 0 -> measure s' < measure s) ->
  step s' = Undefined
  ∨ (∃ t'' : state, step t = Step t'' ∧ R s' t'')
  ∨ (measure s' < measure s ∧ R s' t)
  ∨ (∃ (t'' : state) (m : nat), run (S (S m)) t = Step t'' ∧ R s' t'').
Proof.
  intros Hrun Hrel Hzero. destruct n as [ | [ | m]].
  - simpl in Hrun. injection Hrun as <-. right. right. left.
    split; [ apply Hzero; reflexivity | exact Hrel ].
  - right. left. exists t'. split; [ | exact Hrel ].
    rewrite run_succ in Hrun.
    destruct (step t) as [t0 | | ]; simpl in Hrun; congruence.
  - right. right. right. exists t', m. split; assumption.
Qed.

Lemma alloc_field_kind (d : named) (f : field) (kind : flambda_kind)
    (s : simple) :
  alloc_field d f kind s -> kind = field_kind f.
Proof.
  intros Hf. destruct d as [ | p args | | | | | ]; try contradiction.
  destruct p as [p | p | p | p]; try contradiction.
  - destruct p; try contradiction.
    destruct args as [ | c [ | ]]; try contradiction.
    destruct Hf as (-> & -> & _). reflexivity.
  - destruct p as [[tag] | ]; try contradiction.
    destruct Hf as (i & -> & -> & _). reflexivity.
Qed.

Lemma params_rel_nil_inv (ps ps' : list (variable * flambda_kind)) :
  params_rel M [] ps ps' -> ps = [] ∧ ps' = [].
Proof. intros H. inversion H. split; reflexivity. Qed.

Lemma params_rel_cons_inv (o : option unboxed_fields)
    (sh : list (option unboxed_fields))
    (ps ps' : list (variable * flambda_kind)) :
  params_rel M (o :: sh) ps ps' ->
  ∃ (p : variable) (kind : flambda_kind) ps1, ps = (p, kind) :: ps1.
Proof. intros H. inversion H; subst; eauto. Qed.

Lemma params_rel_none_inv (sh : list (option unboxed_fields))
    (ps ps' : list (variable * flambda_kind)) :
  params_rel M (None :: sh) ps ps' ->
  ∃ (p : variable) (kind : flambda_kind) ps1 ps1',
    ps = (p, kind) :: ps1 ∧ ps' = (p, kind) :: ps1'
    ∧ plain_variable M p = true ∧ params_rel M sh ps1 ps1'.
Proof.
  intros H. inversion H; subst.
  eexists _, _, _, _. repeat split; eassumption.
Qed.

Lemma params_rel_some_inv (sh : list (option unboxed_fields))
    (fields : unboxed_fields) (ps ps' : list (variable * flambda_kind)) :
  params_rel M (Some fields :: sh) ps ps' ->
  ∃ (p : variable) ps1 ps1',
    ps = (p, Value) :: ps1 ∧ ps' = fields_leaves fields ++ ps1'
    ∧ fields_of M p = Some fields ∧ params_rel M sh ps1 ps1'.
Proof.
  intros H. inversion H; subst.
  eexists _, _, _. repeat split; eassumption.
Qed.

Lemma args_rel_nil_inv (args args' : list simple) :
  args_rel M [] args args' ->
  args' = args ∧ forallb (plain_simple M) args = true.
Proof. intros H. inversion H; subst. split; [ reflexivity | assumption ]. Qed.

(* The variables of the fields of an unboxed argument, in the second
   run, hold the fields of its value in the first. *)
Lemma unboxed_args_eval (vars vars' : env) (y : variable)
    (fields_y : unboxed_fields) (w : value) :
  env_rel M vars vars' -> fields_of M y = Some fields_y ->
  lookup y vars = Some w ->
  ∀ (fields : unboxed_fields) (vs : list variable),
    Forall2
      (fun (fu : field * unboxed_field) (v : variable) =>
         (∃ v0 : variable, snd fu = Not_unboxed v0)
         ∧ (fst fu, Not_unboxed v) ∈ fields_y)
      fields vs ->
    ∃ ws : list value,
      eval_simples vars' (map Var vs) = Some ws
      ∧ Forall2
          (fun (fu : field * unboxed_field) (w' : value) =>
             (∃ v : variable, snd fu = Not_unboxed v)
             ∧ field_value (fst fu) w = Some w'
             ∧ kind_of_value w' = field_kind (fst fu))
          fields ws.
Proof.
  intros Hrel Hy Hw. induction 1 as [ | fu v fields vs [Hfu Hin] _ IH].
  - exists []. split; [ reflexivity | constructor ].
  - destruct IH as (ws & Hws & Hall).
    destruct (proj1 (proj2 Hrel) y fields_y w Hy Hw (fst fu) v Hin)
      as (w' & Hfw & Hv & Hk).
    exists (w' :: ws). split.
    + cbn [map eval_simples eval_simple]. rewrite Hv, Hws. reflexivity.
    + constructor; [ | exact Hall ]. split; [ exact Hfu | ].
      split; assumption.
Qed.

Lemma flat_leaves_kinds (w : value) (fields : unboxed_fields)
    (ws : list value) :
  Forall2
    (fun (fu : field * unboxed_field) (w' : value) =>
       (∃ v : variable, snd fu = Not_unboxed v)
       ∧ field_value (fst fu) w = Some w'
       ∧ kind_of_value w' = field_kind (fst fu))
    fields ws ->
  map kind_of_value ws = map snd (fields_leaves fields)
  ∧ ∀ vars : env,
      ∃ vars1 : env, bind_params (leaf_variables fields) ws vars = Some vars1.
Proof.
  induction 1 as [ | [f u] w' fields ws ((v & Hu) & _ & Hk) _ IH].
  - split; [ reflexivity | ]. intros vars. exists vars. reflexivity.
  - simpl in Hu, Hk. subst u. destruct IH as [Hkinds Hbind].
    rewrite leaf_variables_flat. rewrite leaves_cons. simpl.
    rewrite Hk, Hkinds. split; [ reflexivity | ].
    intros vars. exact (Hbind ((v, w') :: vars)).
Qed.

Lemma bind_params_app (l1 l2 : list variable) (vs1 vs2 : list value)
    (vars vars1 : env) :
  bind_params l1 vs1 vars = Some vars1 ->
  bind_params (l1 ++ l2) (vs1 ++ vs2) vars = bind_params l2 vs2 vars1.
Proof.
  revert vs1 vars. induction l1 as [ | x l1 IH]; intros [ | v vs1] vars Hb;
    simpl in Hb |- *; try discriminate Hb.
  - injection Hb as <-. reflexivity.
  - exact (IH vs1 _ Hb).
Qed.

(* The arguments related, given to parameters related, are bound to
   values related. *)
Lemma bind_args_rel (sh : list (option unboxed_fields))
    (args args' : list simple) :
  args_rel M sh args args' ->
  ∀ (ps ps' : list (variable * flambda_kind)) (vars vars' cv cv' hv : env)
    (vs : list value),
    params_rel M sh ps ps' -> env_rel M vars vars' -> env_rel M cv cv' ->
    eval_simples vars args = Some vs ->
    map kind_of_value vs = map snd ps ->
    bind_params (map fst ps) vs cv = Some hv ->
    ∃ (vs' : list value) (hv' : env),
      eval_simples vars' args' = Some vs'
      ∧ map kind_of_value vs' = map snd ps'
      ∧ bind_params (map fst ps') vs' cv' = Some hv' ∧ env_rel M hv hv'.
Proof.
  induction 1 as [args Hargs | o sh | sh a args args' Ha _ IH
                 | sh fields fields_y y vs0 args args' Hy Hvs0 _ IH];
    intros ps ps' vars vars' cv cv' hv vs Hps Hvars Hcv Heval Hkinds Hbind.
  - destruct (params_rel_nil_inv ps ps' Hps) as [-> ->].
    simpl in Hbind. destruct vs as [ | w vs]; [ | discriminate Hbind ].
    injection Hbind as <-.
    destruct args as [ | s args];
      [ | simpl in Heval; destruct (eval_simple vars s);
          [ | discriminate Heval ];
          destruct (eval_simples vars args); discriminate Heval ].
    exists [], cv'. split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ reflexivity | exact Hcv ].
  - simpl in Heval. injection Heval as <-.
    destruct (params_rel_cons_inv o sh ps ps' Hps) as (p & kind & ps1 & ->).
    discriminate Hbind.
  - destruct (params_rel_none_inv sh ps ps' Hps)
      as (p & kind & ps1 & ps1' & -> & -> & Hp & Hps1).
    simpl in Heval.
    destruct (eval_simple vars a) as [w | ] eqn:Hwa; [ | discriminate Heval ].
    destruct (eval_simples vars args) as [vs1 | ] eqn:Hargs;
      [ | discriminate Heval ].
    injection Heval as <-. simpl in Hkinds. injection Hkinds as Hk Hkinds.
    simpl in Hbind.
    destruct (IH ps1 ps1' vars vars' ((p, w) :: cv) ((p, w) :: cv') hv vs1
                Hps1 Hvars (env_rel_bind M cv cv' p w Hcv Hp) Hargs Hkinds
                Hbind)
      as (vs' & hv' & Hev' & Hk' & Hb' & Hrel).
    exists (w :: vs'), hv'. split.
    + simpl. rewrite <- (plain_simple_eval M vars vars' a Hvars Ha), Hwa, Hev'.
      reflexivity.
    + split; [ simpl; rewrite Hk, Hk'; reflexivity | ].
      split; [ exact Hb' | exact Hrel ].
  - destruct (params_rel_some_inv sh fields ps ps' Hps)
      as (p & ps1 & ps1' & -> & -> & Hp & Hps1).
    simpl in Heval.
    destruct (lookup y vars) as [w | ] eqn:Hw; [ | discriminate Heval ].
    destruct (eval_simples vars args) as [vs1 | ] eqn:Hargs;
      [ | discriminate Heval ].
    injection Heval as <-. simpl in Hkinds. injection Hkinds as _ Hkinds.
    simpl in Hbind.
    destruct (unboxed_args_eval vars vars' y fields_y w Hvars Hy Hw fields vs0
                Hvs0)
      as (ws & Hws & Hall).
    destruct (flat_leaves_kinds w fields ws Hall) as [Hwk Hwb].
    destruct (Hwb cv') as (cv1 & Hcv1).
    pose proof (env_rel_bind_unboxed M M_distinct cv cv' cv1 p fields w ws Hcv
                  Hp Hall Hcv1)
      as Hrel1.
    destruct (IH ps1 ps1' vars vars' ((p, w) :: cv) cv1 hv vs1 Hps1 Hvars
                Hrel1 Hargs Hkinds Hbind)
      as (vs' & hv' & Hev' & Hk' & Hb' & Hrel).
    exists (ws ++ vs'), hv'. split.
    + rewrite FlambdaSemanticsProofs.eval_simples_app, Hws, Hev'.
      reflexivity.
    + split; [ rewrite !map_app, Hwk, Hk'; reflexivity | ].
      split; [ | exact Hrel ].
      rewrite map_app. change (map fst (fields_leaves fields))
        with (leaf_variables fields).
      rewrite (bind_params_app _ _ _ _ _ _ Hcv1). exact Hb'.
Qed.

Lemma find_arm_rel (K : finite_map (list (option unboxed_fields))) (i : Z)
    (arms arms' : list (Z * apply_cont)) :
  Forall2
    (fun (arm arm' : Z * apply_cont) =>
       fst arm' = fst arm ∧ ac_rel M K (snd arm) (snd arm'))
    arms arms' ->
  ∀ ac : apply_cont,
    find_arm i arms = Some ac ->
    ∃ ac' : apply_cont, find_arm i arms' = Some ac' ∧ ac_rel M K ac ac'.
Proof.
  induction 1 as [ | [d ac0] [d' ac0'] arms arms' [Hd Hac] _ IH];
    intros ac Hfind; [ discriminate Hfind | ].
  simpl in Hd, Hac, Hfind |- *. subst d'. destruct (Z.eqb i d).
  - injection Hfind as <-. exists ac0'. split; [ reflexivity | exact Hac ].
  - exact (IH ac Hfind).
Qed.

Lemma related_answered (s t : state) :
  related M s t -> answered (related M) eq measure s t.
Proof.
  destruct s as [p vars conts statics hp], t as [q vars' conts' statics' hp'].
  intros (K & Hub & Hvars & Hconts & Hstatics & Hhp).
  cbn [state_expr state_vars state_conts state_statics state_heap] in *.
  subst statics' hp'. unfold answered. revert Hconts.
  destruct Hub as [K x kind d body body' Hx Hd Hbody
    | K x d fields ls body body' Hfields Hxf Hl Hbody
    | K x y fields fields_y l body body' Hfields Hxf Hy Hne Hl Hbody
    | K x kind d f s y fields fields_y nested l body body' Hfields Hxf Hread
        Hs Hy Hne Hin Hl Hbody
    | K x kind d f s y v fields body body' Hx Hread Hs Hfields Hin Hbody
    | K bindings body body' Hbindings Hbody
    | K k sh ps ps' md md' h h' body body' Hps Hmd Hmd' Hh Hbody
    | K S handlers handlers' body body' Hhandlers Hstat Hbody
    | K ac ac' Hac
    | K scrutinee arms arms' Hscrutinee Harms
    | K ]; intros Hconts.
  - (* A [Let] of both. *)
    rewrite step_let, (named_plain_eval M statics vars vars' d Hvars Hd).
    destruct (eval_named statics vars' d) as [w | ] eqn:Hw; [ | exact I ].
    destruct (kind_eqb (kind_of_value w) kind) eqn:Hk; [ | exact I ].
    right. left. eexists. split; [ rewrite step_let, Hw, Hk; reflexivity | ].
    exists K. split; [ exact Hbody | ].
    split; [ exact (env_rel_bind M _ _ x w Hvars Hx) | ].
    split; [ exact Hconts | split; reflexivity ].
  - (* The allocation of an unboxed variable. *)
    rewrite step_let.
    destruct (eval_named statics vars d) as [w | ] eqn:Hw; [ | exact I ].
    destruct (kind_eqb (kind_of_value w) Value) eqn:Hk; [ | exact I ].
    pose proof (alloc_entries2_variables x d fields ls Hl) as Hvl.
    destruct (lets_run (concat ls) body' vars' conts' statics hp)
      as (vars1 & Hrun & Hother & Hbound).
    + rewrite Hvl. exact (leaves_nodup M M_distinct x fields Hfields).
    + intros v kind s v' Hin Hv' ->. rewrite Hvl in Hv'.
      destruct (alloc_entries2_l x d fields ls v kind (Var v') Hl Hin)
        as [(f & _ & _ & Hplain)
           | (f & nested & b & fb & f' & _ & _ & Hb & Hxb & _ & _ & vb & Heq
              & Hinb)].
      * simpl in Hplain. apply plain_variable_iff in Hplain as [_ Hnot].
        pose proof (field_variables_in M x fields v' Hfields Hv') as Hfv.
        apply among_In in Hfv. rewrite Hfv in Hnot. discriminate.
      * injection Heq as <-.
        exact (leaves_apart M M_distinct b x fb fields v' Hb Hfields
                 (not_eq_sym Hxb) (leaf_in fb f' v' Hinb) Hv').
    + intros v kind s Hin.
      destruct (alloc_entries2_l x d fields ls v kind s Hl Hin)
        as [(f & _ & Hf & Hplain)
           | (f & nested & b & fb & f' & _ & Hf & Hb & _ & _ & -> & vb & ->
              & Hinb)].
      * destruct (alloc_field_eval statics vars d w f kind s Hw Hf)
          as (w' & Hs & _ & Hkw).
        exists w'. split; [ | exact Hkw ].
        rewrite <- (plain_simple_eval M vars vars' s Hvars Hplain). exact Hs.
      * destruct (alloc_field_eval statics vars d w f Value (Var b) Hw Hf)
          as (w1 & Hs & _ & _).
        simpl in Hs.
        destruct (proj1 (proj2 Hvars) b fb w1 Hb Hs f' vb Hinb)
          as (w' & _ & Hvb & Hkw).
        exists w'. split; [ exact Hvb | exact Hkw ].
    + apply (run_answer (related M) _ _ _ _ (length (concat ls)) Hrun).
      * exists K. unfold related.
        cbn [state_expr state_vars state_conts state_statics state_heap].
        split; [ exact Hbody | ]. split; [ | split; [ exact Hconts | ] ];
          [ | split; reflexivity ].
        split; [ | split ].
        -- intros z Hz.
           rewrite lookup_other
             by (intros ->; exact (plain_not_unboxed M x fields Hz Hfields)).
           rewrite (proj1 Hvars z Hz). symmetry. apply Hother.
           rewrite Hvl. intros Hzl. apply plain_variable_iff in Hz as [_ Hz].
           pose proof (field_variables_in M x fields z Hfields Hzl) as Hfv.
           apply among_In in Hfv. rewrite Hfv in Hz. discriminate.
        -- intros y fy w0 Hy Hlookup f v Hfv.
           destruct (Pos.eq_dec y x) as [-> | Hne].
           ++ rewrite lookup_head in Hlookup. injection Hlookup as <-.
              rewrite Hfields in Hy. injection Hy as <-.
              destruct (alloc_entries2_flat x d fields ls f v Hl Hfv)
                as (kind & s & Hin & Hf & Hplain).
              destruct (alloc_field_eval statics vars d w f kind s Hw Hf)
                as (w' & Hs & Hfw & Hkw).
              exists w'. split; [ exact Hfw | ].
              split; [ | rewrite Hkw; exact (alloc_field_kind d f kind s Hf) ].
              apply (Hbound v kind s w' Hin).
              rewrite <- (plain_simple_eval M vars vars' s Hvars Hplain).
              exact Hs.
           ++ rewrite lookup_other in Hlookup by exact Hne.
              destruct (proj1 (proj2 Hvars) y fy w0 Hy Hlookup f v Hfv)
                as (w' & Hfw & Hv & Hkv).
              exists w'. split; [ exact Hfw | ].
              split; [ | exact Hkv ]. rewrite Hother; [ exact Hv | ].
              rewrite Hvl. intros Hvx.
              exact (leaves_apart M M_distinct y x fy fields v Hy Hfields Hne
                       (leaf_in fy f v Hfv) Hvx).
        -- intros y fy w0 Hy Hlookup f nested Hfn.
           destruct (Pos.eq_dec y x) as [-> | Hne].
           ++ rewrite lookup_head in Hlookup. injection Hlookup as <-.
              rewrite Hfields in Hy. injection Hy as <-.
              destruct (alloc_entries2_nested x d fields ls f nested Hl Hfn)
                as (b & fb & Hf & Hb & _ & Hall).
              destruct (alloc_field_eval statics vars d w f Value (Var b) Hw
                          Hf)
                as (w1 & Hs & Hfw & _).
              simpl in Hs. exists w1. split; [ exact Hfw | ].
              intros f' v Hv. destruct (Hall f' v Hv) as (vb & Hin & Hinb).
              destruct (proj1 (proj2 Hvars) b fb w1 Hb Hs f' vb Hinb)
                as (w' & Hfw' & Hvb & Hkw).
              exists w'. split; [ exact Hfw' | split; [ | exact Hkw ] ].
              exact (Hbound v (field_kind f') (Var vb) w' Hin Hvb).
           ++ rewrite lookup_other in Hlookup by exact Hne.
              destruct (proj2 (proj2 Hvars) y fy w0 Hy Hlookup f nested Hfn)
                as (w1 & Hw1 & Hn).
              exists w1. split; [ exact Hw1 | ]. intros f' v Hv.
              destruct (Hn f' v Hv) as (w' & Hfw & Hl' & Hkv).
              exists w'. split; [ exact Hfw | split; [ | exact Hkv ] ].
              rewrite Hother; [ exact Hl' | ]. rewrite Hvl. intros Hvx.
              exact (leaves_apart M M_distinct y x fy fields v Hy Hfields Hne
                       (nested_leaf_in fy f nested f' v Hfn Hv) Hvx).
      * intros _. unfold measure. simpl. lia.
  - (* An alias of an unboxed variable. *)
    rewrite step_let. cbn [eval_named eval_simple].
    destruct (lookup y vars) as [w | ] eqn:Hw; [ | exact I ].
    destruct (kind_eqb (kind_of_value w) Value) eqn:Hk; [ | exact I ].
    pose proof (alias_entries_variables fields_y fields l Hl) as Hvl.
    destruct (lets_run l body' vars' conts' statics hp)
      as (vars1 & Hrun & Hother & Hbound).
    + rewrite Hvl. exact (leaves_nodup M M_distinct x fields Hfields).
    + intros v kind s v' Hin Hv' ->.
      destruct (alias_entries_l fields_y fields l v kind (Var v') Hl Hin)
        as (f & _ & _ & v'' & Heq & Hin').
      injection Heq as <-. rewrite Hvl in Hv'.
      exact (leaves_apart M M_distinct y x fields_y fields v' Hy Hfields
               (not_eq_sym Hne) (leaf_in fields_y f v' Hin') Hv').
    + intros v kind s Hin.
      destruct (alias_entries_l fields_y fields l v kind s Hl Hin)
        as (f & _ & -> & v' & -> & Hin').
      destruct (proj1 (proj2 Hvars) y fields_y w Hy Hw f v' Hin')
        as (w' & _ & Hv' & Hkw).
      exists w'. split; [ exact Hv' | exact Hkw ].
    + apply (run_answer (related M) _ _ _ _ (length l) Hrun).
      * exists K. unfold related.
        cbn [state_expr state_vars state_conts state_statics state_heap].
        split; [ exact Hbody | ]. split; [ | split; [ exact Hconts | ] ];
          [ | split; reflexivity ].
        split; [ | split ].
        -- intros z Hz.
           rewrite lookup_other
             by (intros ->; exact (plain_not_unboxed M x fields Hz Hfields)).
           rewrite (proj1 Hvars z Hz). symmetry. apply Hother.
           rewrite Hvl. intros Hzl. apply plain_variable_iff in Hz as [_ Hz].
           pose proof (field_variables_in M x fields z Hfields Hzl) as Hfv.
           apply among_In in Hfv. rewrite Hfv in Hz. discriminate.
        -- intros z fz w0 Hz Hlookup f v Hfv.
           destruct (Pos.eq_dec z x) as [-> | Hne'].
           ++ rewrite lookup_head in Hlookup. injection Hlookup as <-.
              rewrite Hfields in Hz. injection Hz as <-.
              destruct (alias_entries_r fields_y fields l f v Hl Hfv)
                as (kind & s & Hin & -> & v' & -> & Hin').
              destruct (proj1 (proj2 Hvars) y fields_y w Hy Hw f v' Hin')
                as (w' & Hfw & Hv' & Hkw).
              exists w'. split; [ exact Hfw | split; [ | exact Hkw ] ].
              exact (Hbound v (field_kind f) (Var v') w' Hin Hv').
           ++ rewrite lookup_other in Hlookup by exact Hne'.
              destruct (proj1 (proj2 Hvars) z fz w0 Hz Hlookup f v Hfv)
                as (w' & Hfw & Hv & Hkv).
              exists w'. split; [ exact Hfw | ].
              split; [ | exact Hkv ]. rewrite Hother; [ exact Hv | ].
              rewrite Hvl. intros Hvx.
              exact (leaves_apart M M_distinct z x fz fields v Hz Hfields Hne'
                       (leaf_in fz f v Hfv) Hvx).
        -- intros z fz w0 Hz Hlookup f nested Hfn.
           destruct (Pos.eq_dec z x) as [-> | Hne'].
           ++ exfalso. rewrite Hfields in Hz. injection Hz as <-.
              exact (alias_entries_l' fields_y fields l f nested Hl Hfn).
           ++ rewrite lookup_other in Hlookup by exact Hne'.
              destruct (proj2 (proj2 Hvars) z fz w0 Hz Hlookup f nested Hfn)
                as (w1 & Hw1 & Hn).
              exists w1. split; [ exact Hw1 | ]. intros f' v Hv.
              destruct (Hn f' v Hv) as (w' & Hfw & Hl' & Hkv).
              exists w'. split; [ exact Hfw | split; [ | exact Hkv ] ].
              rewrite Hother; [ exact Hl' | ]. rewrite Hvl. intros Hvx.
              exact (leaves_apart M M_distinct z x fz fields v Hz Hfields Hne'
                       (nested_leaf_in fz f nested f' v Hfn Hv) Hvx).
      * intros _. unfold measure. simpl. lia.
  - (* The load of a field unboxed in turn. *)
    rewrite step_let.
    destruct (eval_named statics vars d) as [r | ] eqn:Hr; [ | exact I ].
    destruct (kind_eqb (kind_of_value r) Value) eqn:Hk; [ | exact I ].
    destruct (read_eval statics vars d f s r Hread Hr)
      as (w0 & Hw0 & Hnc & Hfield).
    destruct Hs as [-> | (k & ->)];
      [ | simpl in Hw0; injection Hw0 as <-; exfalso; exact (Hnc _ eq_refl) ].
    simpl in Hw0.
    destruct (proj2 (proj2 Hvars) y fields_y w0 Hy Hw0 f nested Hin)
      as (w & Hw1 & Hn).
    pose proof (Hfield w Hw1) as Hrw. subst r.
    pose proof (alias_entries_variables nested fields l Hl) as Hvl.
    destruct (lets_run l body' vars' conts' statics hp)
      as (vars1 & Hrun & Hother & Hbound).
    + rewrite Hvl. exact (leaves_nodup M M_distinct x fields Hfields).
    + intros v kind0 s v' Hin0 Hv' ->.
      destruct (alias_entries_l nested fields l v kind0 (Var v') Hl Hin0)
        as (f0 & _ & _ & v'' & Heq & Hin').
      injection Heq as <-. rewrite Hvl in Hv'.
      exact (leaves_apart M M_distinct y x fields_y fields v' Hy Hfields
               (not_eq_sym Hne)
               (nested_leaf_in fields_y f nested f0 v' Hin Hin')
               Hv').
    + intros v kind0 s Hin0.
      destruct (alias_entries_l nested fields l v kind0 s Hl Hin0)
        as (f0 & _ & -> & v' & -> & Hin').
      destruct (Hn f0 v' Hin') as (w' & _ & Hv' & Hkw).
      exists w'. split; [ exact Hv' | exact Hkw ].
    + apply (run_answer (related M) _ _ _ _ (length l) Hrun).
      * exists K. unfold related.
        cbn [state_expr state_vars state_conts state_statics state_heap].
        split; [ exact Hbody | ]. split; [ | split; [ exact Hconts | ] ];
          [ | split; reflexivity ].
        split; [ | split ].
        -- intros z Hz.
           rewrite lookup_other
             by (intros ->; exact (plain_not_unboxed M x fields Hz Hfields)).
           rewrite (proj1 Hvars z Hz). symmetry. apply Hother.
           rewrite Hvl. intros Hzl. apply plain_variable_iff in Hz as [_ Hz].
           pose proof (field_variables_in M x fields z Hfields Hzl) as Hfv.
           apply among_In in Hfv. rewrite Hfv in Hz. discriminate.
        -- intros z fz w0' Hz Hlookup f0 v Hfv.
           destruct (Pos.eq_dec z x) as [-> | Hne'].
           ++ rewrite lookup_head in Hlookup. injection Hlookup as <-.
              rewrite Hfields in Hz. injection Hz as <-.
              destruct (alias_entries_r nested fields l f0 v Hl Hfv)
                as (kind0 & s & Hin0 & -> & v' & -> & Hin').
              destruct (Hn f0 v' Hin') as (w' & Hfw & Hv' & Hkw).
              exists w'. split; [ exact Hfw | split; [ | exact Hkw ] ].
              exact (Hbound v (field_kind f0) (Var v') w' Hin0 Hv').
           ++ rewrite lookup_other in Hlookup by exact Hne'.
              destruct (proj1 (proj2 Hvars) z fz w0' Hz Hlookup f0 v Hfv)
                as (w' & Hfw & Hv & Hkv).
              exists w'. split; [ exact Hfw | ].
              split; [ | exact Hkv ]. rewrite Hother; [ exact Hv | ].
              rewrite Hvl. intros Hvx.
              exact (leaves_apart M M_distinct z x fz fields v Hz Hfields Hne'
                       (leaf_in fz f0 v Hfv) Hvx).
        -- intros z fz w0' Hz Hlookup f0 nested0 Hfn.
           destruct (Pos.eq_dec z x) as [-> | Hne'].
           ++ exfalso. rewrite Hfields in Hz. injection Hz as <-.
              exact (alias_entries_l' nested fields l f0 nested0 Hl Hfn).
           ++ rewrite lookup_other in Hlookup by exact Hne'.
              destruct (proj2 (proj2 Hvars) z fz w0' Hz Hlookup f0 nested0 Hfn)
                as (w1 & Hw1' & Hn').
              exists w1. split; [ exact Hw1' | ]. intros f' v Hv.
              destruct (Hn' f' v Hv) as (w' & Hfw & Hl' & Hkv).
              exists w'. split; [ exact Hfw | split; [ | exact Hkv ] ].
              rewrite Hother; [ exact Hl' | ]. rewrite Hvl. intros Hvx.
              exact (leaves_apart M M_distinct z x fz fields v Hz Hfields Hne'
                       (nested_leaf_in fz f0 nested0 f' v Hfn Hv) Hvx).
      * intros _. unfold measure. simpl. lia.
  - (* A read of a field. *)
    rewrite step_let.
    destruct (eval_named statics vars d) as [r | ] eqn:Hr; [ | exact I ].
    destruct (kind_eqb (kind_of_value r) kind) eqn:Hk; [ | exact I ].
    destruct (read_eval statics vars d f s r Hread Hr)
      as (w0 & Hw0 & Hnc & Hfield).
    destruct Hs as [-> | (k & ->)];
      [ | simpl in Hw0; injection Hw0 as <-; exfalso; exact (Hnc _ eq_refl) ].
    simpl in Hw0.
    destruct (proj1 (proj2 Hvars) y fields w0 Hfields Hw0 f v Hin)
      as (w' & Hfw & Hv & _).
    rewrite <- (Hfield w' Hfw) in Hv.
    right. left. eexists. split.
    + rewrite step_let. cbn [eval_named eval_simple]. rewrite Hv, Hk.
      reflexivity.
    + exists K. split; [ exact Hbody | ].
      split; [ exact (env_rel_bind M _ _ x r Hvars Hx) | ].
      split; [ exact Hconts | split; reflexivity ].
  - (* A [Let_static]. *)
    rewrite step_let_static,
      (local_bindings_eval M vars vars' bindings Hvars Hbindings statics).
    destruct (eval_static_bindings vars' bindings statics) as [statics1 | ]
      eqn:Hst; [ | exact I ].
    right. left. eexists.
    split; [ rewrite step_let_static, Hst; reflexivity | ].
    exists K. split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ exact Hconts | split; reflexivity ].
  - (* A non-recursive [Let_cont]. *)
    rewrite step_let_cont_non_recursive. right. left. eexists.
    split; [ rewrite step_let_cont_non_recursive; reflexivity | ].
    exists ((k, sh) :: K). split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ | split; reflexivity ].
    constructor; [ | exact Hconts ].
    exact (cont_rel_closure M sh ps ps' md md' h h' vars vars' K [] conts
             conts' [] [] Hps Hmd Hmd' Hh Hvars Hconts
             (ub_handlers_nil M K)).
  - (* A recursive [Let_cont]. *)
    rewrite step_let_cont_recursive.
    destruct (handlers_bind_statics handlers) eqn:Hb; [ exact I | ].
    right. left. eexists.
    split.
    { rewrite step_let_cont_recursive.
      destruct (handlers_bind_statics handlers') eqn:Hb';
        [ congruence | reflexivity ]. }
    exists (S ++ K). split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ apply rec_closures_rel; assumption | split; reflexivity ].
  - (* An [Apply_cont]. *)
    destruct Hac as (Hk & Htrap & Hargs).
    rewrite step_apply_cont.
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs;
      [ | exact I ].
    unfold apply_continuation.
    destruct (resolve_rel M K conts conts' (ac_continuation ac) Hconts)
      as [(HK & Hc & Hc') | (sh & HK & Hc)].
    + unfold shapes_at in Hargs. rewrite HK in Hargs.
      destruct (args_rel_nil_inv _ _ Hargs) as [Heq Hplain].
      rewrite Hc. cbn [apply_cont_value]. exists vs. split; [ | reflexivity ].
      rewrite step_apply_cont, Heq,
        <- (plain_simples_eval M vars vars' _ Hvars Hplain), Hvs.
      unfold apply_continuation. rewrite Hk, Hc'. reflexivity.
    + unfold shapes_at in Hargs. rewrite HK in Hargs.
      remember (resolve (ac_continuation ac) conts) as c eqn:Ec.
      remember (resolve (ac_continuation ac) conts') as c' eqn:Ec'.
      destruct Hc as [sh ps ps' md md' h h' cvars cvars' Kc S cconts cconts'
                        rec rec' Hps Hmd Hmd' Hh Hcv Hcc Hrec].
      cbn [apply_cont_value]. rewrite Hmd.
      destruct (kinds_eqb (map kind_of_value vs) (map snd ps)) eqn:Hkinds;
        [ | exact I ].
      apply FlambdaSemanticsProofs.kinds_eqb_eq in Hkinds.
      destruct (bind_params (map fst ps) vs cvars) as [hv | ] eqn:Hbind;
        [ | exact I ].
      destruct (bind_args_rel sh _ _ Hargs ps ps' vars vars' cvars cvars' hv vs
                  Hps Hvars Hcv Hvs Hkinds Hbind)
        as (vs' & hv' & Hvs' & Hk' & Hb' & Hrel).
      right. left. eexists. split.
      * rewrite step_apply_cont, Hvs'. unfold apply_continuation.
        rewrite Hk, <- Ec'. cbn [apply_cont_value]. rewrite Hmd', <- Hk'.
        rewrite (proj2 (FlambdaSemanticsProofs.kinds_eqb_eq _ _) eq_refl), Hb'.
        reflexivity.
      * exists (S ++ Kc). split; [ exact Hh | split; [ exact Hrel | ] ].
        split; [ apply rec_closures_rel; assumption | split; reflexivity ].
  - (* A [Switch]. *)
    rewrite step_switch,
      (plain_simple_eval M vars vars' scrutinee Hvars Hscrutinee).
    destruct (eval_simple vars' scrutinee) as [w | ] eqn:Hw; [ | exact I ].
    destruct (naked_immediate_of_value w) as [i | ] eqn:Hi; [ | exact I ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | exact I ].
    destruct (find_arm_rel K i arms arms' Harms ac Harm) as (ac' & Harm' & Hac).
    right. left. eexists.
    split; [ rewrite step_switch, Hw, Hi, Harm'; reflexivity | ].
    exists K. split; [ constructor; exact Hac | ].
    split; [ exact Hvars | split; [ exact Hconts | split; reflexivity ] ].
  - rewrite step_invalid. exact I.
Qed.

(* Related terms, run on their own, refine each other. *)
Lemma ub_refines (p q : expr) :
  ub M [] p q ->
  refines (State p [] [] empty_static_env empty_heap)
    (State q [] [] empty_static_env empty_heap).
Proof.
  intros Hub. apply (shrinking_refines (related M) eq measure).
  - exact related_answered.
  - intros vs vs' ->. reflexivity.
  - exists []. split; [ exact Hub | split; [ apply env_rel_empty | ] ].
    split; [ constructor | split; reflexivity ].
Qed.

End Simulation.

(* ======================================================================== *)
(* Where the unboxed fields come from                                       *)
(* ======================================================================== *)

Section Provenance.

Variable F : facts.
Variable T : tables.
Variable code_my_closures : finite_map variable.
Variable dominated : list (node * node).
Variable to_unbox : list node.

(* The graph reads only fields of blocks and of boxed numbers. *)
Hypothesis reads_blocks :
  forallb (fun '(_, f, _) => is_block_like_field f) (f_accessor F) = true.

Lemma insert_field_in {A : Type} (f : field) (a : A)
    (l : list (field * A)) (x : field * A) :
  x ∈ insert_field f a l -> x = (f, a) ∨ x ∈ l.
Proof.
  induction l as [ | [f' a'] l IH]; simpl;
    [ intros [<- | []]; left; reflexivity | ].
  destruct (field_lt f f'); simpl.
  - intros [<- | Hin]; [ left; reflexivity | right; exact Hin ].
  - intros [<- | Hin]; [ right; left; reflexivity | ].
    destruct (IH Hin) as [-> | Hin']; [ left; reflexivity | right; right ].
    exact Hin'.
Qed.

Lemma sort_fields_in {A : Type} (l : list (field * A)) (x : field * A) :
  x ∈ sort_fields l -> x ∈ l.
Proof.
  unfold sort_fields. induction l as [ | [f a] l IH]; simpl; [ easy | ].
  intros Hin. apply insert_field_in in Hin as [-> | Hin];
    [ left; reflexivity | right; exact (IH Hin) ].
Qed.

(* With no read of a function slot or a call witness, the usages of the
   closures add nothing. *)
Lemma add_usages_none (follow : bool) (out : list node) :
  add_usages_through_function_slots F T code_my_closures follow out = out.
Proof.
  assert (Hnone :
    usages_through_function_slots F T code_my_closures follow out = []).
  { unfold usages_through_function_slots.
    induction out as [ | x out IH]; [ reflexivity | ]. simpl. rewrite IH.
    rewrite app_nil_r. rewrite forallb_forall in reads_blocks.
    assert (Hacc : ∀ l, (∀ a, a ∈ l -> a ∈ f_accessor F) ->
      (if follow
       then
         flat_map
           (fun '(apply_witness, f, base) =>
              if node_eqb base x
                 && field_eqb f (Field_call_witness Known_arity_code_pointer)
              then
                flat_map
                  (fun '(apply_witness', call_witness) =>
                     if node_eqb apply_witness apply_witness'
                     then
                       flat_map
                         (fun '(call_witness', f', code_id) =>
                            if node_eqb call_witness call_witness'
                               && field_eqb f' Field_code_id_of_call_witness
                            then
                              match code_id with
                              | Node_code_id id =>
                                match lookup id code_my_closures with
                                | Some my_closure =>
                                  flat_map
                                    (fun '(my_closure', y) =>
                                       if node_eqb (Node_variable my_closure)
                                            my_closure'
                                          && mem_node y (t_has_usage T)
                                       then [y]
                                       else [])
                                    (t_usages T)
                                | None => []
                                end
                              | _ => []
                              end
                            else [])
                         (f_constructor F)
                     else [])
                  (t_sources T)
              else [])
           l
       else [])
      ++ flat_map
           (fun '(y, f, base) =>
              if node_eqb base x && is_function_slot f
              then
                flat_map
                  (fun '(y', z) =>
                     if node_eqb y y' && mem_node z (t_has_usage T) then [z]
                     else [])
                  (t_usages T)
              else [])
           l
      = []).
    { induction l as [ | [[y f] base] l IHl]; intros Hl;
        [ destruct follow; reflexivity | ].
      pose proof (reads_blocks _ (Hl _ (or_introl eq_refl))) as Hf.
      cbn beta iota in Hf.
      assert (Hrest : ∀ a, a ∈ l -> a ∈ f_accessor F)
        by (intros a Ha; apply Hl; right; exact Ha).
      specialize (IHl Hrest).
      destruct f; try discriminate Hf; destruct follow; simpl;
        rewrite ?andb_false_r; simpl; exact IHl. }
    exact (Hacc (f_accessor F) (fun a Ha => Ha)). }
  unfold add_usages_through_function_slots. simpl. rewrite Hnone.
  reflexivity.
Qed.

Lemma dedup_in (l : list node) (x : node) :
  x ∈ Keyed.dedup node_eqb l -> x ∈ l.
Proof.
  rewrite <- !(keyed_among_in node node_eqb node_eqb_decides).
  rewrite (keyed_among_dedup node node_eqb node_eqb_decides).
  exact (fun H => H).
Qed.

Lemma get_direct_usages_in (x base : node) :
  base ∈ get_direct_usages T [x] ->
  mem_pair x base (t_usages T) = true ∧ mem_node base (t_has_usage T) = true.
Proof.
  unfold get_direct_usages. intros Hin.
  apply dedup_in in Hin.
  apply in_flat_map in Hin as ([x' y] & Hin & Hy).
  simpl in Hy. rewrite orb_false_r in Hy.
  destruct (node_eqb x' x && mem_node y (t_has_usage T)) eqn:Htest;
    [ | destruct Hy ].
  destruct Hy as [<- | []]. apply andb_true_iff in Htest as [Hx Hy].
  apply node_eqb_eq in Hx. subst x'.
  split; [ apply mem_pair_In; exact Hin | exact Hy ].
Qed.

(* A field of [get_fields] is read, from one of the nodes, into a node with
   a use. *)
Lemma get_fields_in (uses : list node) (f : field) (use : field_use) :
  (f, use) ∈ get_fields F T uses ->
  ∃ w base : node,
    (w, f, base) ∈ f_accessor F ∧ mem_node base uses = true
    ∧ (mem_node w (t_any_usage T) = true
       ∨ mem_node w (t_has_usage T) = true).
Proof.
  unfold get_fields. intros Hin. apply sort_fields_in in Hin.
  apply in_flat_map in Hin as (f0 & _ & Hin).
  set (reads := filter
                  (fun '(_, f, base) =>
                     mem_node base uses && negb (is_function_slot f))
                  (f_accessor F)) in Hin.
  assert (Hreads : ∀ w, w ∈ flat_map
                       (fun '(v, f', _) => if field_eqb f0 f' then [v] else [])
                       reads ->
                     ∃ base, (w, f0, base) ∈ f_accessor F
                             ∧ mem_node base uses = true).
  { intros w Hw. apply in_flat_map in Hw as ([[v f'] base] & Hr & Hv).
    destruct (field_eqb f0 f') eqn:Hf; [ | destruct Hv ].
    destruct Hv as [<- | []]. apply field_eqb_eq in Hf. subst f'.
    unfold reads in Hr. apply filter_In in Hr as [Hr Htest].
    apply andb_true_iff in Htest as [Hbase _].
    exists base. split; assumption. }
  destruct (existsb (fun v => mem_node v (t_any_usage T)) _) eqn:Hany.
  - destruct Hin as [[= <- _] | []].
    apply existsb_exists in Hany as (w & Hw & Hau).
    destruct (Hreads w Hw) as (base & Hacc & Hbase).
    exists w, base. split; [ exact Hacc | split; [ exact Hbase | left ] ].
    exact Hau.
  - destruct (Keyed.dedup node_eqb _) as [ | w0 l] eqn:Hdedup;
      [ destruct Hin | ].
    destruct Hin as [[= <- _] | []].
    assert (Hw0 : w0 ∈ Keyed.dedup node_eqb
                    (filter (fun v => mem_node v (t_has_usage T))
                       (flat_map
                          (fun '(v, f', _) =>
                             if field_eqb f0 f' then [v] else [])
                          reads)))
      by (rewrite Hdedup; left; reflexivity).
    apply dedup_in in Hw0.
    apply filter_In in Hw0 as [Hw0 Hhu].
    destruct (Hreads w0 Hw0) as (base & Hacc & Hbase).
    exists w0, base. split; [ exact Hacc | split; [ exact Hbase | right ] ].
    exact Hhu.
Qed.

(* An unboxed field of an allocation is one of the fields given. *)
Lemma mk_unboxed_fields_in (fuel : nat) (block : node) :
  ∀ (fields : list (field * field_use)) (next : positive) (f : field)
    (u : unboxed_field),
    (f, u) ∈ fst (mk_unboxed_fields F T code_my_closures dominated to_unbox
                    fuel block fields next) ->
    ∃ use : field_use, (f, use) ∈ fields.
Proof.
  destruct fuel as [ | fuel]; cbn [mk_unboxed_fields];
    (induction fields as [ | [f' use] fields IH]; intros next f u Hin;
     [ destruct Hin | ]);
    match type of Hin with
    | context [let '(_, _) := ?c in _] => destruct c as [u0 next0]
    end;
    match type of Hin with
    | context [let '(_, _) := ?c in _] => destruct c as [rest next1] eqn:Hrest
    end;
    simpl in Hin;
    (destruct u0 as [u0 | ];
     [ destruct Hin as [[= <- _] | Hin];
       [ exists use; left; reflexivity | ] | ]);
    (destruct (IH next0 f u) as (use' & Huse');
     [ rewrite Hrest; exact Hin | exists use'; right; exact Huse' ]).
Qed.

Lemma unboxed_fields_of_names_in (next : positive) (x : node)
    (fields : unboxed_fields) :
  (x, fields)
    ∈ fst (unboxed_fields_of_names F T code_my_closures dominated to_unbox
             next) ->
  ∃ (a : node) (next' : positive),
    fields
    = fst (mk_unboxed_fields F T code_my_closures dominated to_unbox
             (S (length to_unbox)) a
             (get_fields F T
                (add_usages_through_function_slots F T code_my_closures true
                   (get_direct_usages T [x])))
             next').
Proof.
  unfold unboxed_fields_of_names.
  assert (Hgen : ∀ (l : list (node * node))
                   (acc : list (node * unboxed_fields)) (n : positive),
    (∀ y fs, (y, fs) ∈ acc ->
       ∃ a next',
         fs = fst (mk_unboxed_fields F T code_my_closures dominated to_unbox
                     (S (length to_unbox)) a
                     (get_fields F T
                        (add_usages_through_function_slots F T
                           code_my_closures true (get_direct_usages T [y])))
                     next')) ->
    ∀ y fs,
      (y, fs) ∈ fst (List.fold_left
                       (fun '(acc, next) '(a, x) =>
                          if mem_node a to_unbox
                             && negb (existsb (fun '(y, _) => node_eqb x y)
                                        acc)
                          then
                            let '(fields, next) :=
                              name_unboxed_fields F T code_my_closures
                                dominated to_unbox a x next
                            in
                            (acc ++ [(x, fields)], next)
                          else (acc, next))
                       l (acc, n)) ->
      ∃ a next',
        fs = fst (mk_unboxed_fields F T code_my_closures dominated to_unbox
                    (S (length to_unbox)) a
                    (get_fields F T
                       (add_usages_through_function_slots F T
                          code_my_closures true (get_direct_usages T [y])))
                    next')).
  { induction l as [ | [a x'] l IH]; intros acc n Hacc; [ exact Hacc | ].
    simpl. destruct (mem_node a to_unbox && _); [ | apply IH, Hacc ].
    unfold name_unboxed_fields.
    destruct (mk_unboxed_fields _ _ _ _ _ _ _ _ n) as [fs0 n0] eqn:Hmk.
    apply IH. intros y fs Hin. apply in_app_or in Hin as [Hin | [Heq | []]].
    - exact (Hacc y fs Hin).
    - injection Heq as <- <-. exists a, n. rewrite Hmk. reflexivity. }
  apply (Hgen dominated [] next). intros y fs [].
Qed.

End Provenance.

(* ======================================================================== *)
(* The variables of the fields are fresh                                    *)
(* ======================================================================== *)

Section Fresh.

Variable F : facts.
Variable T : tables.
Variable code_my_closures : finite_map variable.
Variable dominated : list (node * node).
Variable to_unbox : list node.

(* The variables of [fields] are distinct, from [lo] on and before
   [hi]. *)
Definition fresh_range (lo hi : positive) (fields : unboxed_fields) : Prop :=
  (lo <= hi)%positive ∧ NoDup (leaf_variables fields)
  ∧ ∀ v, v ∈ leaf_variables fields -> (lo <= v)%positive ∧ (v < hi)%positive.

Lemma leaf_variables_cons_unboxed (f : field) (nested rest : unboxed_fields) :
  leaf_variables ((f, Unboxed nested) :: rest)
  = leaf_variables nested ++ leaf_variables rest.
Proof. unfold leaf_variables. rewrite leaves_cons, map_app. reflexivity. Qed.

Lemma leaf_variables_cons_var (f : field) (v : variable)
    (rest : unboxed_fields) :
  leaf_variables ((f, Not_unboxed v) :: rest) = v :: leaf_variables rest.
Proof. reflexivity. Qed.

(* A field, unboxed or not, then the rest. *)
Lemma fresh_range_app (lo mid hi : positive) (u : option unboxed_field)
    (f : field) (rest : unboxed_fields) :
  (match u with
   | Some (Not_unboxed v) => lo = v ∧ mid = Pos.succ v
   | Some (Unboxed nested) => fresh_range lo mid nested
   | None => lo = mid
   end) ->
  fresh_range mid hi rest ->
  fresh_range lo hi
    (match u with Some u => (f, u) :: rest | None => rest end).
Proof.
  intros Hu (Hmh & Hnd & Hrest). unfold fresh_range.
  destruct u as [[v | nested] | ]; cbv iota.
  - destruct Hu as [-> ->]. rewrite leaf_variables_cons_var.
    split; [ lia | split ].
    + constructor; [ | exact Hnd ]. intros Hv.
      destruct (Hrest v Hv) as [H1 _]. lia.
    + intros w [<- | Hw]; [ lia | ]. destruct (Hrest w Hw). lia.
  - destruct Hu as (Hlm & Hnd' & Hnested). rewrite leaf_variables_cons_unboxed.
    split; [ lia | split ].
    + apply NoDup_app; [ exact Hnd' | exact Hnd | ].
      intros w Hw Hw'. destruct (Hnested w Hw). destruct (Hrest w Hw'). lia.
    + intros w Hw. apply in_app_or in Hw as [Hw | Hw];
        [ destruct (Hnested w Hw) | destruct (Hrest w Hw) ]; lia.
  - subst mid. split; [ exact Hmh | split; [ exact Hnd | exact Hrest ] ].
Qed.

(* One field of [mk_unboxed_fields], as its inner function makes it. *)
Definition field_step (fuel : nat) (block : node) (f : field) (use : field_use)
    (next : positive) : option unboxed_field * positive :=
  if negb (is_data_field f)
  then (None, next)
  else
    match get_single_field_source F T block f with
    | No_source => (None, next)
    | field_source =>
      let default := (Some (Not_unboxed next), Pos.succ next) in
      match use with
      | Use_unknown => default
      | Use_known flow_to =>
        if forallb (has_to_be_unboxed dominated to_unbox) flow_to
        then
          match field_source, fuel with
          | One_source source, S fuel =>
            let '(nested, next) :=
              mk_unboxed_fields F T code_my_closures dominated to_unbox fuel
                source
                (get_fields F T
                   (add_usages_through_function_slots F T code_my_closures
                      true (get_direct_usages T flow_to)))
                next
            in
            (Some (Unboxed nested), next)
          | _, _ => default
          end
        else default
      end
    end.

Lemma mk_unboxed_fields_cons (fuel : nat) (block : node) (f : field)
    (use : field_use) (fields : list (field * field_use)) (next : positive) :
  mk_unboxed_fields F T code_my_closures dominated to_unbox fuel block
    ((f, use) :: fields) next
  = let '(u, next) := field_step fuel block f use next in
    let '(rest, next) :=
      mk_unboxed_fields F T code_my_closures dominated to_unbox fuel block
        fields next
    in
    (match u with
     | Some u => (f, u) :: rest
     | None => rest
     end,
     next).
Proof. destruct fuel; reflexivity. Qed.

Lemma mk_unboxed_fields_nil (fuel : nat) (block : node) (next : positive) :
  mk_unboxed_fields F T code_my_closures dominated to_unbox fuel block [] next
  = ([], next).
Proof. destruct fuel; reflexivity. Qed.

(* [mk_unboxed_fields] names the variables from [next] on, each once. *)
Lemma mk_unboxed_fields_fresh (fuel : nat) :
  ∀ (block : node) (fields : list (field * field_use)) (next : positive),
    let r := mk_unboxed_fields F T code_my_closures dominated to_unbox fuel
               block fields next in
    fresh_range next (snd r) (fst r).
Proof.
  induction fuel as [ | fuel IHfuel]; intros block fields;
    induction fields as [ | [f use] fields IH]; intros next; cbv zeta;
    rewrite ?mk_unboxed_fields_nil, ?mk_unboxed_fields_cons;
    try (cbn [fst snd];
         split; [ lia | split; [ constructor | intros v [] ] ]).
  - assert (Hstep : let '(u, n) := field_step 0 block f use next in
                    match u with
                    | Some (Not_unboxed v) => next = v ∧ n = Pos.succ v
                    | Some (Unboxed nested) => fresh_range next n nested
                    | None => next = n
                    end).
    { unfold field_step. destruct (negb (is_data_field f)); [ reflexivity | ].
      destruct (get_single_field_source F T block f) as [ | source | ];
        [ reflexivity | | ];
        destruct use as [ | flow_to ]; [ split; reflexivity | |
                                         split; reflexivity | ];
        destruct (forallb _ flow_to); split; reflexivity. }
    destruct (field_step 0 block f use next) as [u n].
    specialize (IH n). cbv zeta in IH.
    destruct (mk_unboxed_fields F T code_my_closures dominated to_unbox 0 block
                fields n) as [rest n1].
    exact (fresh_range_app next n n1 u f rest Hstep IH).
  - assert (Hstep : let '(u, n) := field_step (S fuel) block f use next in
                    match u with
                    | Some (Not_unboxed v) => next = v ∧ n = Pos.succ v
                    | Some (Unboxed nested) => fresh_range next n nested
                    | None => next = n
                    end).
    { unfold field_step. destruct (negb (is_data_field f)); [ reflexivity | ].
      destruct (get_single_field_source F T block f) as [ | source | ];
        [ reflexivity | | ];
        destruct use as [ | flow_to ]; [ split; reflexivity | |
                                         split; reflexivity | ];
        destruct (forallb _ flow_to); try (split; reflexivity).
      pose proof (IHfuel source
                    (get_fields F T
                       (add_usages_through_function_slots F T code_my_closures
                          true (get_direct_usages T flow_to)))
                    next) as Hn.
      cbv zeta in Hn.
      destruct (mk_unboxed_fields _ _ _ _ _ fuel source _ next) as [nested n].
      exact Hn. }
    destruct (field_step (S fuel) block f use next) as [u n].
    specialize (IH n). cbv zeta in IH.
    destruct (mk_unboxed_fields F T code_my_closures dominated to_unbox (S fuel)
                block fields n) as [rest n1].
    exact (fresh_range_app next n n1 u f rest Hstep IH).
Qed.

Lemma field_variables_app (l l' : list (node * unboxed_fields)) :
  field_variables (l ++ l') = field_variables l ++ field_variables l'.
Proof. unfold field_variables. apply flat_map_app. Qed.

Lemma field_variables_one (x : node) (fields : unboxed_fields) :
  field_variables [(x, fields)] = leaf_variables fields.
Proof. unfold field_variables. simpl. rewrite app_nil_r. reflexivity. Qed.

(* The variables of the fields of the names are distinct, from [next] on. *)
Lemma unboxed_fields_of_names_fresh (start : positive) :
  let r := unboxed_fields_of_names F T code_my_closures dominated to_unbox
             start in
  (start <= snd r)%positive ∧ NoDup (field_variables (fst r))
  ∧ ∀ v, v ∈ field_variables (fst r) ->
      (start <= v)%positive ∧ (v < snd r)%positive.
Proof.
  unfold unboxed_fields_of_names.
  assert (Hgen : ∀ (l : list (node * node))
                   (acc : list (node * unboxed_fields)) (n : positive),
    (start <= n)%positive -> NoDup (field_variables acc) ->
    (∀ v, v ∈ field_variables acc ->
       (start <= v)%positive ∧ (v < n)%positive) ->
    let r := List.fold_left
               (fun '(acc, next) '(a, x) =>
                  if mem_node a to_unbox
                     && negb (existsb (fun '(y, _) => node_eqb x y) acc)
                  then
                    let '(fields, next) :=
                      name_unboxed_fields F T code_my_closures dominated
                        to_unbox a x next
                    in
                    (acc ++ [(x, fields)], next)
                  else (acc, next))
               l (acc, n)
    in
    (start <= snd r)%positive ∧ NoDup (field_variables (fst r))
    ∧ ∀ v, v ∈ field_variables (fst r) ->
        (start <= v)%positive ∧ (v < snd r)%positive).
  { induction l as [ | [a x] l IH]; intros acc n Hn Hnd Hin;
      [ split; [ exact Hn | split; [ exact Hnd | exact Hin ] ] | ].
    simpl. destruct (mem_node a to_unbox && _); [ | apply IH; assumption ].
    unfold name_unboxed_fields.
    pose proof (mk_unboxed_fields_fresh (S (length to_unbox)) a
                  (get_fields F T
                     (add_usages_through_function_slots F T code_my_closures
                        true (get_direct_usages T [x])))
                  n) as Hfresh.
    cbv zeta in Hfresh.
    destruct (mk_unboxed_fields _ _ _ _ _ (S (length to_unbox)) a _ n)
      as [fields n'].
    destruct Hfresh as (Hnn' & Hnd' & Hrange). cbn [fst snd] in *.
    apply IH.
    - lia.
    - rewrite field_variables_app, field_variables_one.
      apply NoDup_app; [ exact Hnd | exact Hnd' | ].
      intros v Hv Hv'. destruct (Hin v Hv). destruct (Hrange v Hv'). lia.
    - intros v Hv. rewrite field_variables_app, field_variables_one in Hv.
      apply in_app_or in Hv as [Hv | Hv];
        [ destruct (Hin v Hv) | destruct (Hrange v Hv) ]; lia. }
  apply Hgen; [ lia | constructor | intros v [] ].
Qed.

Lemma fold_max_le (l : list positive) (z : positive) :
  z ∈ l -> (z <= fold_right Pos.max 1 l)%positive.
Proof.
  induction l as [ | a l IH]; intros Hz; [ destruct Hz | ].
  simpl. destruct Hz as [<- | Hz]; [ lia | specialize (IH Hz); lia ].
Qed.

End Fresh.

(* The variables of the fields that the tables give are distinct, and are
   no variable of the unit. *)
Lemma unit_unboxed_fields_fresh (F : facts) (T : tables)
    (dominated : list (node * node)) (to_unbox : list node) (e : expr) :
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  distinct (field_variables unboxed) = true
  ∧ ∀ z : variable,
      z ∈ variables_of e -> among z (field_variables unboxed) = false.
Proof.
  unfold unit_unboxed_fields. cbv zeta.
  destruct (unboxed_fields_of_names_fresh F T (bound_code_my_closures e)
              dominated to_unbox (fresh_name (names_of e)))
    as (_ & Hnd & Hrange).
  split; [ apply distinct_NoDup, Hnd | ].
  intros z Hz. destruct (among z _) eqn:Hamong; [ | reflexivity ].
  apply among_In in Hamong. destruct (Hrange z Hamong) as [Hle _].
  assert (Hnames : z ∈ names_of e)
    by (unfold names_of; apply in_or_app; left; exact Hz).
  apply fold_max_le in Hnames. unfold fresh_name in Hle. lia.
Qed.


(* The checks of FlambdaReaperUnboxingRefinesSpec.v, with the variables of
   the unit fresh, are those above. *)
Section Strong.

Variable unboxed : list (node * unboxed_fields).
Variable sol : solution.

Abbreviation fresh z := (among z (field_variables unboxed) = false).

Lemma strong_variable (x : variable) :
  FlambdaReaperUnboxingRefinesSpec.plain_variable unboxed x = true ->
  fresh x -> plain_variable unboxed x = true.
Proof.
  unfold plain_variable, FlambdaReaperUnboxingRefinesSpec.plain_variable.
  intros -> ->. reflexivity.
Qed.

Lemma strong_simple (s : simple) :
  FlambdaReaperUnboxingRefinesSpec.plain_simple unboxed s = true ->
  (∀ z, z ∈ variables_in_simple s -> fresh z) ->
  plain_simple unboxed s = true.
Proof.
  destruct s as [x | sym | c]; simpl; intros Hs Hz;
    [ | exact Hs | reflexivity ].
  apply strong_variable; [ exact Hs | apply Hz; left; reflexivity ].
Qed.

Lemma strong_simples (l : list simple) :
  forallb (FlambdaReaperUnboxingRefinesSpec.plain_simple unboxed) l = true ->
  (∀ z, z ∈ variables_in_simples l -> fresh z) ->
  forallb (plain_simple unboxed) l = true.
Proof.
  induction l as [ | s l IH]; simpl; [ reflexivity | ].
  rewrite !andb_true_iff. intros [Hs Hl] Hz.
  unfold variables_in_simples in Hz. simpl in Hz.
  split.
  - apply strong_simple; [ exact Hs | ].
    intros z Hin. apply Hz, in_or_app. left. exact Hin.
  - apply IH; [ exact Hl | ]. intros z Hin. apply Hz, in_or_app. right.
    exact Hin.
Qed.

Lemma strong_unboxed_binding (kinds : finite_map flambda_kind)
    (x : variable) (kind : flambda_kind) (d : named)
    (fields : unboxed_fields) :
  FlambdaReaperUnboxingRefinesSpec.unboxed_binding_ok unboxed sol kinds x kind
    d fields
  = true ->
  (∀ z, z ∈ variables_in_named d -> fresh z) ->
  unboxed_binding_ok unboxed sol kinds x kind d fields = true.
Proof.
  unfold unboxed_binding_ok,
    FlambdaReaperUnboxingRefinesSpec.unboxed_binding_ok.
  intros Hd Hz. destruct kind; try discriminate Hd.
  destruct d as [s | p args | | | | | ]; try discriminate Hd;
    [ destruct s; try discriminate Hd; exact Hd | ].
  destruct p as [p | p | p | p]; try discriminate Hd.
  - destruct p; try discriminate Hd;
      destruct args as [ | c [ | ]]; try discriminate Hd; try exact Hd.
    apply andb_true_iff in Hd as [Hc Hfields].
    rewrite (strong_simple c Hc); [ exact Hfields | ].
    intros z Hin. apply Hz. simpl. unfold variables_in_simples. simpl.
    rewrite app_nil_r. exact Hin.
  - destruct p as [[tag] | ]; try discriminate Hd.
    apply forallb_forall. intros [f u] Hin. rewrite forallb_forall in Hd.
    specialize (Hd _ Hin). cbn beta iota in Hd |- *.
    destruct f as [i k | | | | | | | | ]; try discriminate Hd.
    destruct k; try discriminate Hd.
    destruct u as [v | nested]; [ | exact Hd ].
    destruct (nth_error args i) as [a | ] eqn:Hnth; [ | discriminate Hd ].
    apply andb_true_iff in Hd as [Ha Hk]. rewrite Hk, andb_true_r.
    apply (strong_simple a Ha). intros z Hza. apply Hz. simpl.
    apply nth_error_In in Hnth. clear -Hnth Hza.
    induction args as [ | a' args IH]; [ destruct Hnth | ].
    unfold variables_in_simples. simpl. apply in_or_app.
    destruct Hnth as [<- | Hnth]; [ left; exact Hza | right ].
    exact (IH Hnth).
Qed.

Lemma strong_plain_binding (x : variable) (kind : flambda_kind) (d : named) :
  FlambdaReaperUnboxingRefinesSpec.plain_binding_ok unboxed sol x kind d
  = true ->
  (∀ z, z ∈ variables_in_named d -> fresh z) ->
  plain_binding_ok unboxed sol x kind d = true.
Proof.
  unfold plain_binding_ok, FlambdaReaperUnboxingRefinesSpec.plain_binding_ok.
  destruct (unboxed_read unboxed d) as [[f fields] | ]; [ easy | ].
  intros Hd Hz. destruct d as [s | p args | | | | | ]; try discriminate Hd.
  - exact (strong_simple s Hd Hz).
  - exact (strong_simples args Hd Hz).
Qed.

Lemma strong_args_ok (kinds : finite_map flambda_kind) (args : list simple) :
  ∀ params : list (variable * bool),
    FlambdaReaperUnboxingRefinesSpec.args_ok unboxed sol kinds params args
    = true ->
    (∀ z, z ∈ variables_in_simples args -> fresh z) ->
    args_ok unboxed sol kinds params args = true.
Proof.
  induction args as [ | a args IH]; intros params Hok Hz;
    [ destruct params as [ | [? ?] ?]; reflexivity | ].
  unfold variables_in_simples in Hz. simpl in Hz.
  assert (Ha : ∀ z, z ∈ variables_in_simple a -> fresh z)
    by (intros z Hin; apply Hz, in_or_app; left; exact Hin).
  assert (Hargs : ∀ z, z ∈ variables_in_simples args -> fresh z)
    by (intros z Hin; apply Hz, in_or_app; right; exact Hin).
  destruct params as [ | [p keep] params]; simpl in Hok |- *.
  - apply andb_true_iff in Hok as [Ha' Hargs'].
    rewrite (strong_simple a Ha' Ha).
    exact (strong_simples args Hargs' Hargs).
  - apply andb_true_iff in Hok as [Hp Hrest].
    rewrite (IH params Hrest Hargs), andb_true_r.
    destruct (fields_of unboxed p); [ exact Hp | ].
    destruct keep; [ | reflexivity ]. exact (strong_simple a Hp Ha).
Qed.

Lemma strong_apply_cont_ok (kinds : finite_map flambda_kind)
    (conts : finite_map (list (variable * bool))) (ac : apply_cont) :
  FlambdaReaperUnboxingRefinesSpec.apply_cont_ok unboxed sol kinds conts ac
  = true ->
  (∀ z, z ∈ variables_in_simples (ac_args ac) -> fresh z) ->
  apply_cont_ok unboxed sol kinds conts ac = true.
Proof.
  unfold apply_cont_ok, FlambdaReaperUnboxingRefinesSpec.apply_cont_ok.
  intros Hok Hz. destruct (lookup (ac_continuation ac) conts).
  - exact (strong_args_ok kinds _ _ Hok Hz).
  - exact (strong_simples _ Hok Hz).
Qed.

Lemma strong_local :
  (∀ (e : expr) (kinds : finite_map flambda_kind)
     (conts : finite_map (list (variable * bool))),
     FlambdaReaperUnboxingRefinesSpec.local_expr unboxed sol kinds conts e
     = true ->
     (∀ z, z ∈ variables_of e -> fresh z) ->
     local_expr unboxed sol kinds conts e = true)
  ∧ (∀ (h : continuation_handler) (kinds : finite_map flambda_kind)
       (conts : finite_map (list (variable * bool))) (mask : list bool),
       FlambdaReaperUnboxingRefinesSpec.local_handler unboxed sol kinds conts
         mask h
       = true ->
       (∀ z, z ∈ variables_in_handler h -> fresh z) ->
       local_handler unboxed sol kinds conts mask h = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH kinds conts Hloc Hz. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hd Hbody].
    rewrite (Hz x (or_introl eq_refl)). simpl.
    apply andb_true_iff. split.
    + destruct (fields_of unboxed x) as [fields | ].
      * apply (strong_unboxed_binding _ _ _ _ _ Hd).
        intros z Hin. apply Hz. right. apply in_or_app. left. exact Hin.
      * apply (strong_plain_binding _ _ _ Hd).
        intros z Hin. apply Hz. right. apply in_or_app. left. exact Hin.
    + apply (IH _ _ Hbody). intros z Hin. apply Hz. right. apply in_or_app.
      right. exact Hin.
  - intros bindings body _ IH kinds conts Hloc Hz. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hbs Hbody].
    apply andb_true_iff. split.
    + apply forallb_forall. intros b Hb. rewrite forallb_forall in Hbs.
      specialize (Hbs b Hb).
      destruct b as [id c | id | sym [ | | tag fields | | | ]];
        try discriminate Hbs.
      apply (strong_simples fields Hbs). intros z Hin. apply Hz.
      apply in_or_app. left. apply in_flat_map. exists (Bind_symbol sym
        (Static_const_block tag fields)). split; [ exact Hb | exact Hin ].
    + apply (IH _ _ Hbody). intros z Hin. apply Hz. apply in_or_app. right.
      exact Hin.
  - intros k h body IHh IH kinds conts Hloc Hz. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hh Hbody].
    apply andb_true_iff. split.
    + apply (IHh _ _ _ Hh). intros z Hin. apply Hz. apply in_or_app. left.
      exact Hin.
    + apply (IH _ _ Hbody). intros z Hin. apply Hz. apply in_or_app. right.
      exact Hin.
  - intros handlers body IHh IH kinds conts Hloc Hz. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hhs Hbody].
    apply andb_true_iff. split.
    + apply forallb_forall. intros [k h] Hin. rewrite forallb_forall in Hhs.
      apply (IHh k h Hin _ _ _ (Hhs _ Hin)). intros z Hz'. apply Hz.
      apply in_or_app. left. apply in_flat_map. exists (k, h).
      split; [ exact Hin | exact Hz' ].
    + apply (IH _ _ Hbody). intros z Hin. apply Hz. apply in_or_app. right.
      exact Hin.
  - intros ap kinds conts Hloc. discriminate Hloc.
  - intros ac kinds conts Hloc Hz. simpl in Hloc |- *.
    exact (strong_apply_cont_ok _ _ _ Hloc Hz).
  - intros scrutinee arms kinds conts Hloc Hz. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hs Harms].
    apply andb_true_iff. split.
    + apply (strong_simple _ Hs). intros z Hin. apply Hz. apply in_or_app.
      left. exact Hin.
    + apply forallb_forall. intros [d ac] Hin. rewrite forallb_forall in Harms.
      apply (strong_apply_cont_ok _ _ _ (Harms _ Hin)). intros z Hz'.
      apply Hz. apply in_or_app. right. apply in_flat_map. exists (d, ac).
      split; [ exact Hin | exact Hz' ].
  - intros kinds conts _ _. reflexivity.
  - intros params metadata handler IH kinds conts mask Hloc Hz.
    simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hloc Hbody].
    apply andb_true_iff in Hloc as [Hlen Hparams].
    rewrite Hlen. simpl. apply andb_true_iff. split.
    + apply forallb_forall. intros [[p kind] keep] Hp.
      rewrite forallb_forall in Hparams. specialize (Hparams _ Hp).
      unfold param_ok. rewrite Hparams, andb_true_r.
      apply in_combine_l, in_combine_l in Hp.
      apply negb_true_iff. apply Hz. apply in_or_app. left. exact Hp.
    + apply (IH _ _ Hbody). intros z Hin. apply Hz. apply in_or_app. right.
      exact Hin.
Qed.

End Strong.

(* ======================================================================== *)
(* The reads of the graph of a unit without closures                       *)
(* ======================================================================== *)

(* The accessor facts of a list read fields of blocks and of boxed numbers
   only. *)
Definition reads_blocks_only (l : list fact) : Prop :=
  ∀ (to_ : node) (f : field) (base : node),
    Fact_accessor to_ f base ∈ l -> is_block_like_field f = true.

Abbreviation rb a := (reads_blocks_only (acc_facts a)).

Lemma rb_app (l l' : list fact) :
  reads_blocks_only l -> reads_blocks_only l' -> reads_blocks_only (l ++ l').
Proof.
  intros H H' to_ f base Hin. apply in_app_or in Hin as [Hin | Hin];
    [ exact (H _ _ _ Hin) | exact (H' _ _ _ Hin) ].
Qed.

Lemma rb_add_facts (l : list fact) (a : acc) :
  reads_blocks_only l -> rb a -> rb (add_facts l a).
Proof. intros Hl Ha. exact (rb_app _ _ Hl Ha). Qed.

(* Facts none of which is an accessor. *)
Lemma rb_none (l : list fact) :
  (∀ f, f ∈ l -> match f with Fact_accessor _ _ _ => False | _ => True end) ->
  reads_blocks_only l.
Proof. intros H to_ f base Hin. exact (False_rect _ (H _ Hin)). Qed.

Lemma rb_map {A : Type} (g : A -> fact) (l : list A) :
  (∀ a, match g a with Fact_accessor _ _ _ => False | _ => True end) ->
  reads_blocks_only (map g l).
Proof.
  intros H. apply rb_none. intros f Hin.
  apply in_map_iff in Hin as (a & <- & _).
  apply H.
Qed.

Lemma note_simple_facts (s : simple) (a : acc) :
  acc_facts (note_simple s a) = acc_facts a.
Proof. destruct s; reflexivity. Qed.

Lemma note_simples_facts (l : list simple) :
  ∀ a : acc, acc_facts (note_simples l a) = acc_facts a.
Proof.
  unfold note_simples. induction l as [ | s l IH]; intros a;
    [ reflexivity | ].
  simpl. rewrite IH. apply note_simple_facts.
Qed.

Lemma fresh_list_facts (n : nat) :
  ∀ a : acc, acc_facts (snd (fresh_list n a)) = acc_facts a.
Proof.
  induction n as [ | n IH]; intros a; [ reflexivity | ]. simpl.
  destruct (fresh_list n _) as [xs a'] eqn:Hl.
  specialize (IH {| acc_next := Pos.succ (acc_next a);
                    acc_facts := acc_facts a; acc_code_deps := acc_code_deps a;
                    acc_apply_deps := acc_apply_deps a;
                    acc_closure_deps := acc_closure_deps a;
                    acc_symbols := acc_symbols a |}).
  rewrite Hl in IH. exact IH.
Qed.

Lemma rb_cond_any_usage (env : traverse_env) (s : simple) (a : acc) :
  rb a -> rb (cond_any_usage env s a).
Proof.
  intros Ha. unfold cond_any_usage.
  destruct (env_current_code_id env); apply rb_add_facts;
    try (rewrite note_simple_facts; exact Ha);
    apply rb_none; intros f [<- | []]; exact I.
Qed.

Lemma rb_find_cont (env : traverse_env) (k : continuation) (n : nat)
    (a : acc) :
  rb a -> rb (snd (find_cont env k n a)).
Proof.
  intros Ha. unfold find_cont.
  destruct (lookup k (FlambdaReaperTraversal.env_conts env)); [ exact Ha | ].
  pose proof (fresh_list_facts n a) as Hf.
  destruct (fresh_list n a) as [params a']. simpl in Hf |- *.
  apply rb_add_facts; [ apply rb_map; intros; exact I | ].
  rewrite Hf. exact Ha.
Qed.

Lemma rb_block_deps (base : node) (fields : list simple) (a : acc) :
  rb a -> rb (block_deps base fields a).
Proof.
  intros Ha. unfold block_deps. apply rb_add_facts.
  - apply rb_app; [ apply rb_map; intros [i s]; exact I | ].
    apply rb_none. intros f [<- | [<- | []]]; exact I.
  - rewrite note_simples_facts. exact Ha.
Qed.

Lemma rb_traverse_prim (env : traverse_env) (x : variable) (p : prim)
    (args : list simple) (a : acc) :
  rb a -> rb (traverse_prim env x p args a).
Proof.
  intros Ha. unfold traverse_prim.
  assert (Hacc : ∀ f s a, is_block_like_field f = true -> rb a ->
            rb (add_fact (Fact_accessor (Node_variable x) f
                            (node_of_simple s)) (note_simple s a))).
  { intros f s a0 Hf Ha0. apply rb_add_facts.
    - intros to_ f' base [Heq | []]. injection Heq as _ <- _. exact Hf.
    - rewrite note_simple_facts. exact Ha0. }
  assert (Hdefault :
    rb (add_facts
          (Fact_any_source (Node_variable x)
           :: map (fun from => Fact_use (Node_variable x) from)
                (free_names_of_simples args))
          (if prim_has_arbitrary_effects p
           then cond_any_usage env (Var x) a else a))).
  { apply rb_add_facts.
    - apply (rb_app [_]); [ apply rb_none; intros f [<- | []]; exact I | ].
      apply rb_map. intros; exact I.
    - destruct (prim_has_arbitrary_effects p);
        [ apply rb_cond_any_usage, Ha | exact Ha ]. }
  cbv zeta.
  destruct p as [p | p | p | p]; [ | exact Hdefault | | ].
  - destruct p; destruct args as [ | s [ | s' args]]; try exact Hdefault;
      try destruct mut; try destruct variant_only; cbv beta iota;
      first
        [ exact Hdefault
        | apply Hacc; [ reflexivity | exact Ha ]
        | apply rb_add_facts;
          [ apply rb_none; intros f [<- | []]; exact I | ];
          first
            [ apply Hacc; [ reflexivity | exact Ha ]
            | rewrite note_simple_facts; exact Ha ] ].
  - destruct p; [ apply rb_block_deps, Ha | exact Hdefault ].
  - destruct p; try exact Hdefault. apply rb_block_deps, Ha.
Qed.

Lemma rb_apply_cont_deps (env : traverse_env) (ac : apply_cont) (a : acc) :
  rb a -> rb (apply_cont_deps env ac a).
Proof.
  intros Ha. unfold apply_cont_deps.
  pose proof (rb_find_cont env (ac_continuation ac) (length (ac_args ac)) a Ha)
    as Hf.
  destruct (find_cont _ _ _ a) as [params a']. simpl in Hf.
  apply rb_add_facts; [ apply rb_map; intros [p s]; exact I | ].
  rewrite note_simples_facts. exact Hf.
Qed.

Lemma rb_traverse_switch (env : traverse_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (a : acc) :
  rb a -> rb (traverse_switch env scrutinee arms a).
Proof.
  intros Ha. unfold traverse_switch.
  pose proof (rb_cond_any_usage env scrutinee a Ha) as H.
  revert H. generalize (cond_any_usage env scrutinee a).
  induction (Keyed.compact Z.eqb arms) as [ | [d ac] l IH]; intros a' Ha';
    [ exact Ha' | ].
  simpl. apply IH. apply rb_apply_cont_deps, Ha'.
Qed.

Lemma rb_exn_handler_deps (env : traverse_env) (h : continuation_handler)
    (a : acc) :
  rb a -> rb (exn_handler_deps env h a).
Proof.
  intros Ha. unfold exn_handler_deps.
  destruct h as [[ | p params] metadata handler]; [ exact Ha | ].
  destruct (hm_is_exn_handler metadata); [ | exact Ha ].
  apply rb_cond_any_usage, rb_add_facts; [ | exact Ha ].
  apply rb_none. intros f [<- | []]. exact I.
Qed.

(* A unit without closures, projections, calls and code. *)
Lemma rb_traverse :
  (∀ e : expr, closure_free e = true ->
     ∀ (env : traverse_env) (a : acc), rb a -> rb (traverse env e a))
  ∧ (∀ h : continuation_handler, handler_closure_free h = true ->
       ∀ (env : traverse_env) (a : acc), rb a ->
         rb (traverse_handler env h a)).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH Hcf env a Ha. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hd Hbody].
    change (traverse env (Let x kind d body) a)
      with (traverse env body (traverse_named env x d a)).
    apply (IH Hbody).
    destruct d as [s | p args | | | | | ]; try discriminate Hd.
    + apply rb_add_facts; [ apply rb_none; intros f [<- | []]; exact I | ].
      rewrite note_simple_facts. exact Ha.
    + apply rb_traverse_prim, Ha.
  - intros bindings body _ IH Hcf env a Ha. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hbs Hbody].
    change (traverse env (Let_static bindings body) a)
      with (traverse env body
              (List.fold_left (fun a b => traverse_binding env b a) bindings
                 (List.fold_left (fun a b => traverse_static_binding env b a)
                    bindings
                    (List.fold_left (fun a b => prepare_binding b a)
                       bindings a)))).
    apply (IH Hbody).
    rewrite forallb_forall in Hbs.
    assert (Hblocks : ∀ b, b ∈ bindings ->
              ∃ s tag fields,
                b = Bind_symbol s (Static_const_block tag fields)).
    { intros b Hb. specialize (Hbs b Hb).
      destruct b as [ | | s [ | | tag fields | | | ]]; try discriminate Hbs.
      eauto. }
    clear Hbs.
    assert (Hprep : ∀ a,
              List.fold_left (fun a b => prepare_binding b a) bindings a = a).
    { clear -Hblocks. induction bindings as [ | b bindings IHb]; intros a;
        [ reflexivity | ].
      destruct (Hblocks b (or_introl eq_refl)) as (s & tag & fields & ->).
      simpl. apply IHb. intros b' Hb'. apply Hblocks. right. exact Hb'. }
    assert (Hcode : ∀ a,
              List.fold_left (fun a b => traverse_binding env b a) bindings a
              = a).
    { clear -Hblocks. induction bindings as [ | b bindings IHb]; intros a;
        [ reflexivity | ].
      destruct (Hblocks b (or_introl eq_refl)) as (s & tag & fields & ->).
      simpl. apply IHb. intros b' Hb'. apply Hblocks. right. exact Hb'. }
    rewrite Hcode, Hprep. clear Hprep Hcode.
    revert a Ha. induction bindings as [ | b bindings IHb]; intros a Ha;
      [ exact Ha | ].
    destruct (Hblocks b (or_introl eq_refl)) as (s & tag & fields & ->).
    simpl. apply IHb; [ intros b' Hb'; apply Hblocks; right; exact Hb' | ].
    apply rb_block_deps, Ha.
  - intros k h body IHh IH Hcf env a Ha. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hh Hbody].
    change (traverse env (Let_cont_non_recursive k h body) a)
      with (traverse (add_cont env k (handler_params h)) body
              (exn_handler_deps env h (traverse_handler env h a))).
    apply (IH Hbody).
    apply rb_exn_handler_deps. exact (IHh Hh env a Ha).
  - intros handlers body IHh IH Hcf env a Ha. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hhs Hbody].
    change (traverse env (Let_cont_recursive handlers body) a)
      with (traverse (add_rec_conts env handlers) body
              (List.fold_left
                 (fun a '(_, h) =>
                    traverse_handler (add_rec_conts env handlers) h a)
                 handlers a)).
    apply (IH Hbody).
    rewrite forallb_forall in Hhs.
    revert a Ha. generalize (add_rec_conts env handlers) as env'.
    intros env'. assert (Hsub : ∀ k h, (k, h) ∈ handlers -> (k, h) ∈ handlers)
      by auto. revert Hsub. generalize handlers at 1 3 as hs.
    intros hs Hsub. induction hs as [ | [k h] hs IHhs]; intros a Ha;
      [ exact Ha | ].
    simpl. apply IHhs; [ intros k' h' Hin; apply Hsub; right; exact Hin | ].
    assert (Hin : (k, h) ∈ handlers) by (apply Hsub; left; reflexivity).
    exact (IHh k h Hin (Hhs _ Hin) env' a Ha).
  - intros ap Hcf. discriminate Hcf.
  - intros ac _ env a Ha. apply rb_apply_cont_deps, Ha.
  - intros scrutinee arms _ env a Ha. apply rb_traverse_switch, Ha.
  - intros _ env a Ha. exact Ha.
  - intros params metadata handler IH Hcf env a Ha. exact (IH Hcf env a Ha).
Qed.

Lemma rb_closure_deps (cds : finite_map code_dep) (ds : list closure_dep) :
  ∀ a : acc,
    rb a -> rb (List.fold_left (fun a d => add_closure_dep cds d a) ds a).
Proof.
  induction ds as [ | d ds IH]; intros a0 Ha0; [ exact Ha0 | ].
  simpl. apply IH. unfold add_closure_dep.
  destruct (lookup (cld_closure_code_id d) cds) as [cd | ].
  - apply rb_add_facts; [ | exact Ha0 ].
    apply rb_none. intros f Hf.
    destruct Hf as [<- | [<- | Hf]]; [ exact I | exact I | ].
    destruct (cd_unknown_arity_call_witnesses cd); [ destruct Hf | ].
    destruct Hf as [<- | []]. exact I.
  - destruct (fresh a0) as [witness a1] eqn:Hfresh.
    apply rb_add_facts.
    + apply rb_none. intros f Hf.
      repeat (destruct Hf as [<- | Hf]; [ exact I | ]). destruct Hf.
    + unfold fresh in Hfresh. injection Hfresh as _ <-. exact Ha0.
Qed.

Lemma rb_resolve_delayed_deps (symbols : list symbol) (a : acc) :
  rb a -> rb (resolve_delayed_deps symbols a).
Proof.
  intros Ha. unfold resolve_delayed_deps.
  apply rb_add_facts; [ apply rb_map; intros; exact I | ].
  apply rb_closure_deps. apply rb_add_facts; [ | exact Ha ].
  apply rb_none. intros f Hf. apply in_flat_map in Hf as (d & _ & Hf).
  unfold apply_dep_facts in Hf.
  destruct (lookup _ _); [ unfold alias_for_caller in Hf | ];
    destruct (ad_function_containing_apply_expr d);
    repeat (destruct Hf as [<- | Hf]; [ exact I | ]); destruct Hf.
Qed.

Lemma rb_f_accessor (l : list fact) :
  reads_blocks_only l ->
  forallb (fun '(_, f, _) => is_block_like_field f)
    (f_accessor (facts_of_list l))
  = true.
Proof.
  intros Hl. apply forallb_forall. intros [[to_ f] base] Hin.
  cbn [facts_of_list f_accessor] in Hin.
  apply in_flat_map in Hin as (fct & Hfct & Hin).
  destruct fct; simpl in Hin; try contradiction.
  destruct Hin as [[= <- <- <-] | []]. exact (Hl _ _ _ Hfct).
Qed.

(* The graph of a unit without closures, projections, calls and code reads
   only fields of blocks and of boxed numbers. *)
Lemma closure_free_reads_blocks (flag : reaper_preserve_direct_calls)
    (u : FlambdaWellFormedSpec.unit_interface) (e : expr) :
  closure_free e = true ->
  forallb (fun '(_, f, _) => is_block_like_field f)
    (f_accessor (traverse_unit flag u e))
  = true.
Proof.
  intros Hcf. unfold traverse_unit. cbv zeta.
  apply rb_f_accessor. apply rb_resolve_delayed_deps.
  apply rb_add_facts; [ apply rb_none; intros f [<- | []]; exact I | ].
  apply (proj1 rb_traverse e Hcf).
  apply rb_add_facts; [ apply rb_none; intros f [<- | []]; exact I | ].
  apply rb_add_facts; [ apply rb_none; intros f [<- | []]; exact I | ].
  intros to_ f base [].
Qed.

(* The fragment has no closure, projection, call or code. *)
Lemma local_closure_free (unboxed : list (node * unboxed_fields))
    (sol : solution) :
  (∀ (e : expr) (kinds : finite_map flambda_kind)
     (conts : finite_map (list (variable * bool))),
     FlambdaReaperUnboxingRefinesSpec.local_expr unboxed sol kinds conts e
     = true
     -> closure_free e = true)
  ∧ (∀ (h : continuation_handler) (kinds : finite_map flambda_kind)
       (conts : finite_map (list (variable * bool))) (mask : list bool),
       FlambdaReaperUnboxingRefinesSpec.local_handler unboxed sol kinds conts
         mask h
       = true ->
       handler_closure_free h = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH kinds conts Hloc. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hd Hbody].
    rewrite (IH _ _ Hbody), andb_true_r.
    destruct (fields_of unboxed x) as [fields | ].
    + unfold FlambdaReaperUnboxingRefinesSpec.unboxed_binding_ok in Hd.
      destruct kind; try discriminate Hd.
      destruct d; try discriminate Hd; reflexivity.
    + unfold FlambdaReaperUnboxingRefinesSpec.plain_binding_ok in Hd.
      destruct (unboxed_read unboxed d) as [[f fields] | ] eqn:Hread.
      * unfold unboxed_read in Hread. destruct d; try discriminate Hread.
        reflexivity.
      * destruct d; try discriminate Hd; reflexivity.
  - intros bindings body _ IH kinds conts Hloc. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hbs Hbody].
    rewrite (IH _ _ Hbody), andb_true_r. apply forallb_forall. intros b Hb.
    rewrite forallb_forall in Hbs. specialize (Hbs b Hb).
    destruct b as [ | | s [ | | | | | ]]; try discriminate Hbs. reflexivity.
  - intros k h body IHh IH kinds conts Hloc. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hh Hbody].
    rewrite (IHh _ _ _ Hh), (IH _ _ Hbody). reflexivity.
  - intros handlers body IHh IH kinds conts Hloc. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hhs Hbody].
    rewrite (IH _ _ Hbody), andb_true_r. apply forallb_forall.
    intros [k h] Hin. rewrite forallb_forall in Hhs.
    exact (IHh k h Hin _ _ _ (Hhs _ Hin)).
  - intros ap kinds conts Hloc. discriminate Hloc.
  - intros ac kinds conts _. reflexivity.
  - intros scrutinee arms kinds conts _. reflexivity.
  - intros kinds conts _. reflexivity.
  - intros params metadata handler IH kinds conts mask Hloc.
    simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [_ Hbody]. exact (IH _ _ Hbody).
Qed.

(* ======================================================================== *)
(* The two rebuilds of a unit of the fragment                               *)
(* ======================================================================== *)

Section Rebuilds.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable preserve_direct_calls : bool.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable first_fresh first_fresh_u : positive.

(* What the graph has of the unit: the facts of FlambdaReaperFlowProofs.v
   ([expr_edges]), and a solution closed under the rules. *)
Variable noted : symbol -> Prop.
Variable deps : closure_dep -> Prop.
Variable cds : finite_map code_dep.
Variable kept_slots : value_slot -> bool.
Variable codes0 : list (code_id * code).
Variable symbols : list symbol.
Variable is_local : field -> bool.
Hypothesis sol_closed : closed is_local (graph_of_facts F) sol.
Hypothesis local_slots : is_local_slots_only is_local.
(* Each unboxed field is read, from a use of the variable, into a node
   that has a use ([get_fields]). *)
Hypothesis fields_read : ∀ (x : variable) (fields : unboxed_fields)
    (f : field) (u : unboxed_field),
  fields_of unboxed x = Some fields -> (f, u) ∈ fields ->
  ∃ w base : node,
    g_accessor (graph_of_facts F) w f base
    ∧ s_usages sol (Node_variable x) base = true
    ∧ s_has_usage sol w = true.

Abbreviation edges ce e :=
  (FlambdaReaperFlowProofs.expr_edges (graph_of_facts F) noted deps cds
     kept_slots codes0 symbols True None ce e).

(* A simple that [rewrite_simple] keeps. *)
Definition kept_simple (en : rebuild_env) (s : simple) : bool :=
  match s with
  | Var x => is_var_used sol x (kind_of_variable en x)
  | Symbol sym => is_symbol_used sol sym
  | Const _ => true
  end.

(* A field unboxed is used, and what the allocation holds there is kept:
   the solution has it of the field read. *)
Lemma field_kept (x : variable) (fields : unboxed_fields) (f : field)
    (v : unboxed_field) (a : simple) (en : rebuild_env) :
  fields_of unboxed x = Some fields -> (f, v) ∈ fields ->
  g_constructor (graph_of_facts F) (Node_variable x) f (node_of_simple a) ->
  is_block_like_field f = true ->
  s_field_used sol (Node_variable x) f = true ∧ kept_simple en a = true.
Proof.
  intros Hx Hin Hcons Hblock.
  destruct (fields_read x fields f v Hx Hin) as (w & base & Hacc & Hus & Hw).
  assert (Hlocal : is_local f = false).
  { destruct (is_local f) eqn:Hl; [ | reflexivity ].
    destruct (local_slots f Hl) as [(w' & ->) | (fs & ->)];
      discriminate Hblock. }
  split.
  - exact (field_used_usage _ _ _ sol_closed _ _ _ _ _ Hcons Hus Hacc Hw).
  - assert (Hhu : s_has_usage sol (node_of_simple a) = true).
    { destruct (s_any_usage sol (Node_variable x)) eqn:Hau.
      - apply (has_usage_any _ _ _ sol_closed).
        exact (any_usage_constructor _ _ _ sol_closed _ _ _ Hau Hcons Hlocal).
      - apply (has_usage_alias _ _ _ sol_closed w); [ exact Hw | ].
        exact (alias_constructor_usage _ _ _ sol_closed _ _ _ _ _ Hcons Hus
                 Hau Hacc). }
    destruct a as [y | sym | c]; simpl in Hhu |- *; [ | | reflexivity ].
    + unfold is_var_used. destruct (kind_of_variable en y); try reflexivity;
        exact Hhu.
    + exact Hhu.
Qed.

Abbreviation sol_u := (solution_without_unboxed sol unboxed).

Definition plain_rebuild (en : rebuild_env) (e : expr) : expr :=
  rebuild_expr F sol preserve_direct_calls unit_code_ids first_fresh en e.

Definition unbox_rebuild (en : unbox_env) (e : expr) : expr :=
  rebuild_expr_u F sol cannot_change code_deps preserve_direct_calls
    unit_code_ids codes unboxed first_fresh_u [] en e.

Definition is_decided (d : decision) : bool :=
  match d with
  | Decided _ => true
  | Unbox _ => false
  end.

(* The handlers in scope as the fragment has them ([C]): their
   parameters, with what the plain rebuild keeps of each. The plain
   rebuild has their masks ([pen_of]), the rebuild with unboxing their
   decisions ([conts_match]), and [ub] the shapes of the parameters that
   the plain rebuild keeps ([K_of]). *)
Definition pen_of (en : unbox_env) (C : finite_map (list (variable * bool)))
    : rebuild_env :=
  {| env_conts := map (fun '(k, ps) => (k, map snd ps)) C;
     env_kinds := env_kinds (ue_env en);
     env_projections := [] |}.

Definition shape_of_param (pk : variable * bool)
    : list (option unboxed_fields) :=
  let '(p, keep) := pk in if keep then [fields_of unboxed p] else [].

Definition K_of (C : finite_map (list (variable * bool)))
    : finite_map (list (option unboxed_fields)) :=
  map (fun '(k, ps) => (k, flat_map shape_of_param ps)) C.

Definition decision_match (pk : variable * bool) (d : decision) : Prop :=
  let '(p, keep) := pk in
  match fields_of unboxed p with
  | Some fields => d = Unbox fields ∧ (keep = true ∨ fields = [])
  | None =>
    (d = Decided Delete ∧ keep = false)
    ∨ ((∃ kind : flambda_kind, d = Decided (Keep kind)) ∧ keep = true)
  end.

Definition conts_match (C : finite_map (list (variable * bool)))
    (D : finite_map (list decision)) : Prop :=
  Forall2
    (fun (kp : continuation * list (variable * bool))
         (kd : continuation * list decision) =>
       fst kd = fst kp ∧ Forall2 decision_match (snd kp) (snd kd))
    C D.

(* No projection is pending. *)
Record uinv (en : unbox_env) : Prop := {
  uinv_drop : ue_drop_projections en = false;
  uinv_projections : env_projections (ue_env en) = []
}.


Lemma plain_not_bound (s : simple) :
  plain_simple unboxed s = true -> unboxed_of unboxed s = None.
Proof.
  destruct s as [x | sym | c]; simpl; intros Hs; [ | | reflexivity ].
  - apply plain_variable_iff in Hs as [Hs _]. unfold Keyed.is_bound in Hs.
    unfold unboxed_of_node. destruct (Keyed.lookup _ _ _); congruence.
  - apply negb_true_iff in Hs. unfold Keyed.is_bound in Hs.
    unfold unboxed_of_node. destruct (Keyed.lookup _ _ _); congruence.
Qed.

Lemma rewrite_simple_u (en : rebuild_env) (s : simple) :
  plain_simple unboxed s = true ->
  rewrite_simple sol_u en s = rewrite_simple sol en s.
Proof.
  destruct s as [x | sym | c]; simpl; intros Hs; [ | | reflexivity ].
  - apply plain_variable_iff in Hs as [Hs _].
    unfold is_var_used. destruct (kind_of_variable en x); try reflexivity;
      cbn [solution_without_unboxed s_has_usage is_unboxed_name];
      rewrite Hs, andb_true_r; reflexivity.
  - apply negb_true_iff in Hs. unfold is_symbol_used.
    cbn [solution_without_unboxed s_has_usage is_unboxed_name].
    rewrite Hs, andb_true_r. reflexivity.
Qed.

(* What the plain rebuild makes of a plain simple is plain. *)
Lemma rewrite_simple_plain (en : rebuild_env) (s : simple) :
  plain_simple unboxed s = true ->
  plain_simple unboxed (rewrite_simple sol en s) = true.
Proof.
  destruct s as [x | sym | c]; simpl; intros Hs; [ | | reflexivity ].
  - destruct (is_var_used sol x _); [ exact Hs | reflexivity ].
  - destruct (is_symbol_used sol sym); [ exact Hs | reflexivity ].
Qed.

Lemma rewrite_simples_plain (en : rebuild_env) (args : list simple) :
  forallb (plain_simple unboxed) args = true ->
  forallb (plain_simple unboxed) (map (rewrite_simple sol en) args) = true.
Proof.
  induction args as [ | s args IH]; simpl; [ reflexivity | ].
  rewrite !andb_true_iff. intros [Hs Hargs].
  split; [ apply rewrite_simple_plain, Hs | apply IH, Hargs ].
Qed.

Lemma rebuild_fields_u (en : rebuild_env) (base : node) (args : list simple) :
  forallb (plain_simple unboxed) args = true ->
  ∀ i : nat,
    rebuild_fields sol_u en base i args = rebuild_fields sol en base i args.
Proof.
  induction args as [ | s args IH]; simpl; [ reflexivity | ].
  rewrite andb_true_iff. intros [Hs Hargs] i.
  rewrite (rewrite_simple_u en s Hs), (IH Hargs). reflexivity.
Qed.

Lemma rebuild_fields_plain (en : rebuild_env) (base : node)
    (args : list simple) :
  forallb (plain_simple unboxed) args = true ->
  ∀ i : nat,
    forallb (plain_simple unboxed) (rebuild_fields sol en base i args) = true.
Proof.
  induction args as [ | s args IH]; simpl; [ reflexivity | ].
  rewrite andb_true_iff. intros [Hs Hargs] i. rewrite andb_true_iff.
  split; [ | apply IH, Hargs ].
  destruct (field_used sol base _); [ apply rewrite_simple_plain, Hs | ].
  reflexivity.
Qed.

Lemma rebuild_fields_nth (en : rebuild_env) (base : node)
    (args : list simple) :
  ∀ (k i : nat) (s : simple),
    nth_error args i = Some s ->
    nth_error (rebuild_fields sol en base k args) i
    = Some (if field_used sol base (Field_block (k + i) Value)
            then rewrite_simple sol en s
            else Const (Poison Value)).
Proof.
  induction args as [ | a args IH]; intros k i s Hnth;
    [ destruct i; discriminate Hnth | ].
  destruct i as [ | i]; simpl in Hnth |- *.
  - injection Hnth as ->. rewrite Nat.add_0_r. reflexivity.
  - rewrite (IH (S k) i s Hnth). rewrite Nat.add_succ_r. reflexivity.
Qed.

(* The plain rebuild of a [Let] that is not of an unboxed variable, nor a
   read of one. *)
Lemma named_plain_u (en : rebuild_env) (x : variable) (d : named) :
  named_plain unboxed d = true ->
  rebuild_named sol_u en x d = rebuild_named sol en x d.
Proof.
  intros Hd. destruct d as [s | p args | | | | | ]; simpl in Hd;
    try discriminate Hd.
  - simpl. rewrite (rewrite_simple_u en s Hd). reflexivity.
  - assert (Hmap : map (rewrite_simple sol_u en) args
                   = map (rewrite_simple sol en) args).
    { apply map_ext_in. intros s Hs. apply rewrite_simple_u.
      rewrite forallb_forall in Hd. exact (Hd s Hs). }
    destruct p as [p | p | p | p]; destruct p; simpl; rewrite ?Hmap;
      rewrite ?(rebuild_fields_u en (Node_variable x) args Hd 0);
      reflexivity.
Qed.

Lemma named_plain_rebuilt (en : rebuild_env) (x : variable) (d : named) :
  named_plain unboxed d = true ->
  named_plain unboxed (rebuild_named sol en x d) = true.
Proof.
  intros Hd. destruct d as [s | p args | | | | | ]; simpl in Hd;
    try discriminate Hd.
  - simpl. apply rewrite_simple_plain, Hd.
  - destruct p as [p | p | p | p]; simpl;
      try (apply rewrite_simples_plain, Hd).
    + destruct p; simpl; try (apply rewrite_simples_plain, Hd).
      destruct (field_used sol _ _); simpl;
        [ apply rewrite_simples_plain, Hd | reflexivity ].
    + destruct p; simpl; try (apply rewrite_simples_plain, Hd).
      apply rebuild_fields_plain, Hd.
    + destruct p; simpl; try (apply rewrite_simples_plain, Hd).
      apply rebuild_fields_plain, Hd.
Qed.

Lemma lets_free_continuations (l : list (variable * flambda_kind * simple))
    : ∀ body : expr,
      free_continuations_of_expr (lets l body) = free_continuations_of_expr body
      ∧ binds_statics (lets l body) = binds_statics body.
Proof.
  induction l as [ | [[v kind] s] l IH]; intros body; [ split; reflexivity | ].
  change (lets ((v, kind, s) :: l) body)
    with (lets l (Let v kind (Simple s) body)).
  destruct (IH (Let v kind (Simple s) body)) as [H1 H2].
  rewrite H1, H2. split; reflexivity.
Qed.

Lemma kinds_env_kept (en : rebuild_env) (s : simple) :
  kept_simple (kinds_env (env_kinds en)) s = kept_simple en s.
Proof. destruct s; reflexivity. Qed.

Lemma kinds_env_kind (en : rebuild_env) (s : simple) :
  simple_kind (kinds_env (env_kinds en)) s = simple_kind en s.
Proof. destruct s; reflexivity. Qed.

Lemma kept_rewrite (en : rebuild_env) (s : simple) :
  kept_simple en s = true -> rewrite_simple sol en s = s.
Proof.
  destruct s as [x | sym | c]; simpl; intros H; [ | | reflexivity ];
    rewrite H; reflexivity.
Qed.

(* The fields of a block that is unboxed. *)
Lemma bind_block_fields_lets (en : rebuild_env) (x : variable) (tag : Z)
    (args : list simple) (fields : unboxed_fields) :
  forallb (plain_simple unboxed) args = true ->
  (∀ (f : field) (u : unboxed_field),
     (f, u) ∈ fields ->
     ∃ (i : nat) (v : variable) (a : simple),
       f = Field_block i Value ∧ u = Not_unboxed v
       ∧ nth_error args i = Some a
       ∧ s_field_used sol (Node_variable x) f = true
       ∧ kept_simple en a = true ∧ simple_kind en a = Value) ->
  ∀ body : expr,
    ∃ l : list (variable * flambda_kind * simple),
      bind_block_fields unboxed en tag args fields body = Some (lets l body)
      ∧ alloc_entries unboxed
          (Prim (Variadic (Make_block (Block_kind_values tag)))
             (rebuild_fields sol en (Node_variable x) 0 args))
          fields l.
Proof.
  intros Hargs. unfold bind_block_fields.
  induction fields as [ | [f u] fields IH]; intros Hfields body.
  - exists []. split; [ reflexivity | constructor ].
  - destruct (Hfields f u (or_introl eq_refl))
      as (i & v & a & -> & -> & Hnth & Hused & Hkept & Hkind).
    assert (Ha : plain_simple unboxed a = true).
    { rewrite forallb_forall in Hargs. apply Hargs.
      exact (nth_error_In _ _ Hnth). }
    destruct (IH (fun f' u' Hin => Hfields f' u' (or_intror Hin))
                (Let v Value (Simple a) body)) as (l & Hl & Hentries).
    exists ((v, Value, a) :: l). split.
    + simpl List.fold_left. rewrite Hnth, Hkind, kind_eqb_refl.
      rewrite (plain_not_bound a Ha). exact Hl.
    + constructor; [ | exact Hentries ].
      split; [ reflexivity | split; [ | exact Ha ] ].
      exists i. split; [ reflexivity | split; [ reflexivity | ] ].
      rewrite (rebuild_fields_nth en _ args 0 i a Hnth). simpl.
      unfold field_used. rewrite Hused, (kept_rewrite en a Hkept).
      reflexivity.
Qed.

(* The field of a boxed number that is unboxed. *)
Lemma bind_boxed_number_lets (bn : boxable_number) (c : simple)
    (fields : unboxed_fields) (d : named) :
  plain_simple unboxed c = true ->
  (fields <> [] -> d = Prim (Unary (Box_number bn)) [c]) ->
  (∀ (f : field) (u : unboxed_field),
     (f, u) ∈ fields ->
     ∃ v : variable, f = Field_boxed_number bn ∧ u = Not_unboxed v) ->
  ∀ body : expr,
    ∃ l : list (variable * flambda_kind * simple),
      bind_boxed_number_field bn c fields body = Some (lets l body)
      ∧ alloc_entries unboxed d fields l.
Proof.
  intros Hc Hd. unfold bind_boxed_number_field.
  induction fields as [ | [f u] fields IH]; intros Hfields body.
  - exists []. split; [ reflexivity | constructor ].
  - destruct (Hfields f u (or_introl eq_refl)) as (v & -> & ->).
    assert (Hd' : d = Prim (Unary (Box_number bn)) [c])
      by (apply Hd; discriminate).
    destruct (IH (fun _ => Hd') (fun f' u' Hin => Hfields f' u' (or_intror Hin))
                (Let v (unboxed_kind bn) (Simple c) body))
      as (l & Hl & Hentries).
    exists ((v, unboxed_kind bn, c) :: l). split.
    + simpl List.fold_left. rewrite Nat.eqb_refl. exact Hl.
    + constructor; [ | exact Hentries ]. subst d.
      split; [ reflexivity | split; [ | exact Hc ] ].
      split; [ reflexivity | split; reflexivity ].
Qed.

Lemma with_kinds_combine (params : list variable)
    (kinds : list flambda_kind) :
  length params = length kinds ->
  with_kinds params kinds = combine params kinds.
Proof.
  revert kinds. induction params as [ | p params IH]; intros [ | k kinds] Hlen;
    simpl in Hlen |- *; try discriminate Hlen; [ reflexivity | ].
  rewrite IH by lia. reflexivity.
Qed.

Lemma params_to_keep_length (is_exn : bool)
    (bound : list (variable * flambda_kind)) :
  length (params_to_keep sol is_exn bound) = length bound.
Proof. unfold params_to_keep. apply length_map. Qed.

(* The arguments of an [Apply_cont]. *)
Lemma get_args_decided (en : rebuild_env) (args : list simple) :
  forallb (plain_simple unboxed) args = true ->
  ∀ ds : list decision,
    forallb is_decided ds = true ->
    get_args sol unboxed en ds args
    = Some (filter_kept (map is_kept_decision ds)
              (map (rewrite_simple sol en) args)).
Proof.
  induction args as [ | a args IH]; intros Hargs ds Hds; [ reflexivity | ].
  simpl in Hargs. apply andb_true_iff in Hargs as [Ha Hargs].
  destruct ds as [ | d ds].
  - simpl. rewrite (IH Hargs [] eq_refl), (rewrite_simple_u en a Ha).
    reflexivity.
  - simpl in Hds. apply andb_true_iff in Hds as [Hd Hds].
    simpl. rewrite (IH Hargs ds Hds).
    destruct d as [[kind | ] | fields]; simpl in Hd; try discriminate Hd;
      simpl; [ rewrite (rewrite_simple_u en a Ha) | ]; reflexivity.
Qed.

Lemma field_eqb_decides : decides_equality field_eqb.
Proof.
  intros f f'. split; [ apply field_eqb_eq | intros <-; apply field_eqb_refl ].
Qed.

Lemma map_fst_combine {A B : Type} (l : list A) (l' : list B) :
  length l = length l' -> map fst (combine l l') = l.
Proof.
  revert l'. induction l as [ | a l IH]; intros [ | b l'] Hlen;
    simpl in Hlen |- *; try discriminate Hlen; [ reflexivity | ].
  rewrite IH by lia. reflexivity.
Qed.

Lemma map_snd_combine {A B : Type} (l : list A) (l' : list B) :
  length l = length l' -> map snd (combine l l') = l'.
Proof.
  revert l'. induction l as [ | a l IH]; intros [ | b l'] Hlen;
    simpl in Hlen |- *; try discriminate Hlen; [ reflexivity | ].
  rewrite IH by lia. reflexivity.
Qed.

Lemma filter_kept_forallb {A : Type} (p : A -> bool) (mask : list bool)
    (l : list A) :
  forallb p l = true -> forallb p (filter_kept mask l) = true.
Proof.
  revert mask. induction l as [ | a l IH]; intros mask Hl; [ reflexivity | ].
  simpl in Hl. apply andb_true_iff in Hl as [Ha Hl].
  destruct mask as [ | [ | ] mask]; simpl; rewrite ?Ha, ?IH by exact Hl;
    try reflexivity; apply IH, Hl.
Qed.

Lemma lookup_map_values {A B : Type} (f : A -> B) (k : positive)
    (l : list (positive * A)) :
  lookup k (map (fun '(k', a) => (k', f a)) l) = option_map f (lookup k l).
Proof.
  induction l as [ | [k' a] l IH]; [ reflexivity | ]. cbn [map].
  rewrite !lookup_cons. destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

(* The [Let] of an unboxed variable, as the fragment has it. *)
Lemma unboxed_binding_cases (kinds : finite_map flambda_kind) (x : variable)
    (kind : flambda_kind) (d : named) (fields : unboxed_fields) :
  unboxed_binding_ok unboxed sol kinds x kind d fields = true ->
  kind = Value
  ∧ ((∃ (tag : Z) (args : list simple),
        d = Prim (Variadic (Make_block (Block_kind_values tag))) args
        ∧ ∀ (f : field) (u : unboxed_field),
            (f, u) ∈ fields ->
            ∃ (i : nat) (a : simple),
              f = Field_block i Value ∧ nth_error args i = Some a
              ∧ simple_kind (kinds_env kinds) a = Value
              ∧ (((∃ v : variable, u = Not_unboxed v)
                  ∧ plain_simple unboxed a = true)
                 ∨ ∃ (nested : unboxed_fields) (b : variable)
                     (fields_b : unboxed_fields),
                     u = Unboxed nested ∧ a = Var b ∧ x <> b
                     ∧ fields_of unboxed b = Some fields_b
                     ∧ fields_alike nested fields_b = true))
     ∨ (∃ (bn : boxable_number) (c : simple),
          d = Prim (Unary (Box_number bn)) [c]
          ∧ plain_simple unboxed c = true
          ∧ ∀ (f : field) (u : unboxed_field),
              (f, u) ∈ fields ->
              ∃ v : variable, f = Field_boxed_number bn ∧ u = Not_unboxed v)
     ∨ (∃ (y : variable) (fields_y : unboxed_fields),
          d = Simple (Var y) ∧ x <> y ∧ fields_of unboxed y = Some fields_y
          ∧ forallb
              (fun '(f, u) =>
                 match u, Keyed.lookup field_eqb f fields_y with
                 | Not_unboxed _, Some (Not_unboxed _) => true
                 | _, _ => false
                 end)
              fields
            = true
          ∧ is_var_used sol y (kind_of_variable (kinds_env kinds) y) = true)
     ∨ (∃ (bk : block_access_kind) (mut : mutability) (i : Z) (y : variable)
          (fields_y nested : unboxed_fields),
          d = Prim (Unary (Block_load bk mut i)) [Var y] ∧ x <> y
          ∧ fields_of unboxed y = Some fields_y
          ∧ Keyed.lookup field_eqb (Field_block (Z.to_nat i) Value) fields_y
            = Some (Unboxed nested)
          ∧ fields_alike fields nested = true)).
Proof.
  unfold unboxed_binding_ok. intros H.
  destruct kind; try discriminate H. split; [ reflexivity | ].
  destruct d as [s | p args | | | | | ]; try discriminate H.
  { destruct s as [y | | ]; try discriminate H.
    right. right. left. apply andb_true_iff in H as [Hne H].
    unfold unboxed_arg_ok in H.
    destruct (fields_of unboxed y) as [fields_y | ] eqn:Hy;
      [ | discriminate H ].
    apply andb_true_iff in H as [Hflat Hused].
    exists y, fields_y. split; [ reflexivity | ].
    split; [ intros ->; rewrite Pos.eqb_refl in Hne; discriminate Hne | ].
    split; [ exact Hy | split; [ exact Hflat | exact Hused ] ]. }
  destruct p as [p | p | p | p]; try discriminate H.
  - destruct p; try discriminate H.
    + destruct args as [ | [y | | ] [ | ]]; try discriminate H.
      right. right. right. apply andb_true_iff in H as [Hne H].
      destruct (fields_of unboxed y) as [fields_y | ] eqn:Hy;
        [ | discriminate H ].
      destruct (Keyed.lookup field_eqb _ fields_y) as [[ | nested] | ] eqn:Hn;
        try discriminate H.
      eexists _, _, _, y, fields_y, nested. split; [ reflexivity | ].
      split; [ intros ->; rewrite Pos.eqb_refl in Hne; discriminate Hne | ].
      split; [ exact Hy | split; [ exact Hn | exact H ] ].
    + rename kind into bn.
      destruct args as [ | c [ | ]]; try discriminate H.
      right. left. apply andb_true_iff in H as [Hc Hfields].
      exists bn, c. split; [ reflexivity | split; [ exact Hc | ] ].
      intros f u Hin. rewrite forallb_forall in Hfields.
      specialize (Hfields (f, u) Hin). cbn beta iota in Hfields.
      destruct u as [v | ]; [ | discriminate Hfields ].
      apply field_eqb_eq in Hfields. subst f.
      exists v. split; reflexivity.
  - destruct p as [[tag] | ]; try discriminate H.
    left. exists tag, args. split; [ reflexivity | ].
    intros f u Hin. rewrite forallb_forall in H.
    specialize (H (f, u) Hin). cbn beta iota in H.
    destruct f as [i k | | | | | | | | ]; try discriminate H.
    destruct k; try discriminate H.
    destruct u as [v | nested].
    + destruct (nth_error args i) as [a | ] eqn:Hnth; [ | discriminate H ].
      apply andb_true_iff in H as [Ha Hk]. apply kind_eqb_eq in Hk.
      exists i, a. split; [ reflexivity | split; [ exact Hnth | ] ].
      split; [ exact Hk | ]. left. split; [ exists v; reflexivity | exact Ha ].
    + destruct (nth_error args i) as [[b | | ] | ] eqn:Hnth;
        try discriminate H.
      apply andb_true_iff in H as [H Hok].
      apply andb_true_iff in H as [Hne Hk]. apply kind_eqb_eq in Hk.
      unfold unboxed_arg_ok in Hok.
      destruct (fields_of unboxed b) as [fields_b | ] eqn:Hb;
        [ | discriminate Hok ].
      apply andb_true_iff in Hok as [Halike _].
      exists i, (Var b). split; [ reflexivity | split; [ exact Hnth | ] ].
      split; [ exact Hk | ]. right. exists nested, b, fields_b.
      split; [ reflexivity | split; [ reflexivity | ] ].
      split; [ intros ->; rewrite Pos.eqb_refl in Hne; discriminate Hne | ].
      split; [ exact Hb | exact Halike ].
Qed.

(* The [Let] of a variable that is not unboxed, as the fragment has
   it. *)
Lemma plain_binding_cases (x : variable) (kind : flambda_kind) (d : named) :
  plain_binding_ok unboxed sol x kind d = true ->
  named_plain unboxed d = true
  ∨ ∃ (p : unary_primitive) (y : variable) (f : field)
      (fields : unboxed_fields),
      d = Prim (Unary p) [Var y]
      ∧ read_of (Prim (Unary p) [Var y]) = Some (f, Var y)
      ∧ (∀ s : simple, read_of (Prim (Unary p) [s]) = Some (f, s))
      ∧ (∀ en : rebuild_env,
           rebuild_named sol en x d
           = Prim (Unary p) [rewrite_simple sol en (Var y)])
      ∧ fields_of unboxed y = Some fields
      ∧ (is_var_used sol x kind = true ->
         ∃ v : variable,
           Keyed.lookup field_eqb f fields = Some (Not_unboxed v)).
Proof.
  unfold plain_binding_ok. destruct (unboxed_read unboxed d) as [[f fields] | ]
    eqn:Hread.
  - intros H. right. unfold unboxed_read in Hread.
    destruct d as [ | prim args | | | | | ]; try discriminate Hread.
    destruct prim as [p | | | ]; try discriminate Hread.
    destruct args as [ | [y | | ] [ | ]]; try discriminate Hread;
      destruct p; try discriminate Hread;
      simpl in Hread;
      destruct (fields_of unboxed y) as [fs | ] eqn:Hy; try discriminate Hread;
      injection Hread as <- <-;
      eexists _, y, _, fs;
      (split; [ reflexivity | split; [ reflexivity | split;
        [ intros s; reflexivity | split; [ intros en; reflexivity | ] ] ] ]);
      (split; [ exact Hy | ]);
      intros Hused; rewrite Hused in H; simpl in H;
      destruct (Keyed.lookup field_eqb _ fs) as [[v | ] | ]; try discriminate H;
      exists v; reflexivity.
  - intros H. left. destruct d; exact H.
Qed.

Lemma pen_projections (en : unbox_env)
    (C : finite_map (list (variable * bool))) :
  env_projections (pen_of en C) = [].
Proof. reflexivity. Qed.

Lemma rewrite_simple_kinds (en1 en2 : rebuild_env) (s : simple) :
  env_kinds en1 = env_kinds en2 ->
  rewrite_simple sol en1 s = rewrite_simple sol en2 s.
Proof.
  intros H. destruct s; simpl; [ | reflexivity | reflexivity ].
  unfold kind_of_variable. rewrite H. reflexivity.
Qed.

Lemma rebuild_fields_kinds (en1 en2 : rebuild_env) (base : node)
    (args : list simple) :
  env_kinds en1 = env_kinds en2 ->
  ∀ i : nat,
    rebuild_fields sol en1 base i args = rebuild_fields sol en2 base i args.
Proof.
  intros H. induction args as [ | s args IH]; intros i; simpl;
    [ reflexivity | ].
  rewrite (rewrite_simple_kinds en1 en2 s H), IH. reflexivity.
Qed.

Lemma rebuild_named_kinds (en1 en2 : rebuild_env) (x : variable)
    (d : named) :
  env_kinds en1 = env_kinds en2 ->
  rebuild_named sol en1 x d = rebuild_named sol en2 x d.
Proof.
  intros H.
  assert (Hm : ∀ l, map (rewrite_simple sol en1) l
                    = map (rewrite_simple sol en2) l)
    by (intros l; apply map_ext; intros s; apply rewrite_simple_kinds, H).
  destruct d as [s | p args | | | | | ]; simpl;
    rewrite ?(rewrite_simple_kinds en1 en2 _ H); try reflexivity.
  destruct p as [p | p | p | p]; destruct p; simpl; rewrite ?Hm;
    rewrite ?(rebuild_fields_kinds en1 en2 _ args H 0); reflexivity.
Qed.

Lemma uinv_bind (en : unbox_env) (bound : list (variable * flambda_kind)) :
  uinv (ubind_variables en bound).
Proof. split; reflexivity. Qed.

Lemma uinv_continuations (en : unbox_env)
    (bound : list (continuation * list decision)) :
  uinv (ubind_continuations en bound).
Proof. split; reflexivity. Qed.

Lemma conts_match_lookup (C : finite_map (list (variable * bool)))
    (D : finite_map (list decision)) :
  conts_match C D ->
  ∀ k : continuation,
    (lookup k C = None ∧ lookup k D = None)
    ∨ ∃ (ps : list (variable * bool)) (ds : list decision),
        lookup k C = Some ps ∧ lookup k D = Some ds
        ∧ Forall2 decision_match ps ds.
Proof.
  induction 1 as [ | [k' ps] [k'' ds] C D [Hk Hds] _ IH]; intros k;
    [ left; split; reflexivity | ].
  simpl in Hk, Hds. subst k''. rewrite !lookup_cons.
  destruct (Pos.eqb k k').
  - right. exists ps, ds. split; [ reflexivity | split; [ reflexivity | ] ].
    exact Hds.
  - exact (IH k).
Qed.

(* The variables of the fields of an argument unboxed alike. *)
Lemma unboxed_args_flat (fields : unboxed_fields) (y : variable)
    (fields_y : unboxed_fields) :
  fields_of unboxed y = Some fields_y ->
  forallb
    (fun '(f, u) =>
       match u, Keyed.lookup field_eqb f fields_y with
       | Not_unboxed _, Some (Not_unboxed _) => true
       | _, _ => false
       end)
    fields
  = true ->
  ∃ (l : list (simple * flambda_kind)) (vs : list variable),
    unboxed_args unboxed fields (Var y) = Some l ∧ map fst l = map Var vs
    ∧ Forall2
        (fun (fu : field * unboxed_field) (v : variable) =>
           (∃ v0 : variable, snd fu = Not_unboxed v0)
           ∧ (fst fu, Not_unboxed v) ∈ fields_y)
        fields vs.
Proof.
  intros Hy Hf. unfold unboxed_args. cbn [unboxed_of].
  change (unboxed_of_node unboxed (Node_variable y))
    with (fields_of unboxed y).
  rewrite Hy.
  enough (H : ∃ (pairs : list (variable * variable * flambda_kind))
                (vs : list variable),
             unboxed_field_pairs Value (Unboxed fields) (Unboxed fields_y)
             = Some pairs
             ∧ map (fun '(_, x, _) => x) pairs = vs
             ∧ Forall2
                 (fun (fu : field * unboxed_field) (v : variable) =>
                    (∃ v0 : variable, snd fu = Not_unboxed v0)
                    ∧ (fst fu, Not_unboxed v) ∈ fields_y)
                 fields vs).
  { destruct H as (pairs & vs & -> & Hvs & Hall).
    exists (map (fun '(_, x, kind) => (Var x, kind)) pairs), vs.
    split; [ reflexivity | split; [ | exact Hall ] ].
    rewrite <- Hvs, !map_map. apply map_ext. intros [[a b] c]. reflexivity. }
  induction fields as [ | [f u] fields IH]; simpl in Hf.
  - exists [], []. split; [ reflexivity | split; [ reflexivity | ] ].
    constructor.
  - apply andb_true_iff in Hf as [Hfu Hf].
    destruct (IH Hf) as (pairs & vs & Hp & Hvs & Hall).
    destruct u as [v0 | nested]; [ | discriminate Hfu ].
    destruct (Keyed.lookup field_eqb f fields_y) as [[v | ] | ] eqn:Hl;
      try discriminate Hfu.
    exists ((v0, v, field_kind f) :: pairs), (v :: vs).
    split; [ | split ].
    + change (unboxed_field_pairs Value
                (Unboxed ((f, Not_unboxed v0) :: fields))
                (Unboxed fields_y))
        with (match Keyed.lookup field_eqb f fields_y with
              | Some u' =>
                match unboxed_field_pairs (field_kind f) (Not_unboxed v0) u',
                      unboxed_field_pairs Value (Unboxed fields)
                        (Unboxed fields_y)
                with
                | Some p, Some q => Some (p ++ q)
                | _, _ => None
                end
              | None => None
              end).
      rewrite Hl, Hp. reflexivity.
    + simpl. rewrite Hvs. reflexivity.
    + constructor; [ | exact Hall ]. split; [ exists v0; reflexivity | ].
      exact (keyed_lookup_in field_eqb field_eqb_decides f _ fields_y Hl).
Qed.

(* The arguments of an [Apply_cont] to a handler of the unit. *)
Lemma get_args_rel (en : rebuild_env) (args : list simple) :
  ∀ (ps : list (variable * bool)) (ds : list decision),
    Forall2 decision_match ps ds ->
    args_ok unboxed sol (env_kinds en) ps args = true ->
    ∃ args_u : list simple,
      get_args sol unboxed en ds args = Some args_u
      ∧ args_rel unboxed (flat_map shape_of_param ps)
          (filter_kept (map snd ps) (map (rewrite_simple sol en) args))
          args_u.
Proof.
  induction args as [ | a args IH]; intros ps ds Hds Hok.
  - exists []. split; [ reflexivity | ]. cbn [map].
    replace (filter_kept (map snd ps) []) with (@nil simple)
      by (destruct (map snd ps); reflexivity).
    destruct (flat_map shape_of_param ps) as [ | o sh ];
      [ apply args_rel_beyond; reflexivity | apply args_rel_missing ].
  - destruct Hds as [ | [p keep] d ps ds Hd Hds ].
    + exists (filter_kept [] (map (rewrite_simple sol en) (a :: args))).
      split; [ exact (get_args_decided en (a :: args) Hok [] eq_refl) | ].
      apply args_rel_beyond. apply filter_kept_forallb, rewrite_simples_plain.
      exact Hok.
    + simpl in Hok. apply andb_true_iff in Hok as [Ha Hok].
      destruct (IH ps ds Hds Hok) as (rest & Hrest & Hrel).
      cbn [get_args]. rewrite Hrest.
      unfold decision_match in Hd.
      cbn [flat_map shape_of_param map filter_kept].
      destruct (fields_of unboxed p) as [fields | ] eqn:Hp.
      * destruct Hd as [-> Hkeep].
        unfold unboxed_arg_ok in Ha.
        destruct a as [y | sym | c]; try discriminate Ha.
        destruct (fields_of unboxed y) as [fields_y | ] eqn:Hy;
          [ | discriminate Ha ].
        apply andb_true_iff in Ha as [Hflat Hused].
        destruct (unboxed_args_flat fields y fields_y Hy Hflat)
          as (l & vs & Hl & Hlvs & Hall).
        rewrite Hl. cbn [option_map]. rewrite Hlvs.
        exists (map Var vs ++ rest). split; [ reflexivity | ].
        destruct keep.
        -- cbn [app]. simpl in Hused. unfold kinds_env in Hused.
           cbn [rewrite_simple]. unfold kind_of_variable in Hused |- *.
           cbn [env_kinds] in Hused. rewrite Hused.
           exact (args_rel_unboxed unboxed _ fields fields_y y vs _ _ Hy Hall
                    Hrel).
        -- destruct Hkeep as [Hk | ->]; [ discriminate Hk | ].
           inversion Hall. subst vs. exact Hrel.
      * destruct Hd as [[-> ->] | [(kind & ->) ->]].
        -- exists rest. split; [ reflexivity | exact Hrel ].
        -- simpl in Ha. exists (rewrite_simple sol_u en a :: rest).
           split; [ reflexivity | ].
           rewrite (rewrite_simple_u en a Ha).
           apply args_rel_plain; [ apply rewrite_simple_plain, Ha | ].
           exact Hrel.
Qed.

Lemma rewrite_apply_cont_rel (en : unbox_env)
    (C : finite_map (list (variable * bool))) (ac : apply_cont) :
  conts_match C (ue_decisions en) ->
  apply_cont_ok unboxed sol (env_kinds (ue_env en)) C ac = true ->
  (rewrite_apply_cont sol (pen_of en C) ac = None
   ∧ rewrite_apply_cont_u sol unboxed en ac = None)
  ∨ ∃ ac_p ac_u : apply_cont,
      rewrite_apply_cont sol (pen_of en C) ac = Some ac_p
      ∧ rewrite_apply_cont_u sol unboxed en ac = Some ac_u
      ∧ ac_rel unboxed (K_of C) ac_p ac_u.
Proof.
  intros HC Hok. unfold rewrite_apply_cont, rewrite_apply_cont_u.
  destruct (existsb (simple_has_no_source sol) (ac_args ac));
    [ left; split; reflexivity | right ].
  unfold apply_cont_ok in Hok.
  assert (Hmap : map (rewrite_simple sol (pen_of en C)) (ac_args ac)
                 = map (rewrite_simple sol (ue_env en)) (ac_args ac))
    by (apply map_ext; intros s; apply rewrite_simple_kinds; reflexivity).
  rewrite Hmap. cbn [pen_of env_conts]. rewrite lookup_map_values.
  destruct (conts_match_lookup C _ HC (ac_continuation ac))
    as [[HCk HDk] | (ps & ds & HCk & HDk & Hds)];
    rewrite HCk in Hok |- *; rewrite HDk; cbn [option_map].
  - assert (Hu : map (rewrite_simple sol_u (ue_env en)) (ac_args ac)
                 = map (rewrite_simple sol (ue_env en)) (ac_args ac)).
    { apply map_ext_in. intros s Hs. apply rewrite_simple_u.
      rewrite forallb_forall in Hok. exact (Hok s Hs). }
    rewrite Hu. eexists _, _. split; [ reflexivity | split; [ reflexivity | ] ].
    unfold ac_rel, shapes_at, K_of.
    cbn [ac_continuation ac_args ac_trap_action].
    rewrite lookup_map_values, HCk. cbn [option_map].
    split; [ reflexivity | split; [ reflexivity | ] ].
    apply args_rel_beyond, rewrite_simples_plain, Hok.
  - destruct (get_args_rel (ue_env en) (ac_args ac) ps ds Hds Hok)
      as (args_u & Hget & Hrel).
    rewrite Hget. eexists _, _.
    split; [ reflexivity | split; [ reflexivity | ] ].
    unfold ac_rel, shapes_at, K_of.
    cbn [ac_continuation ac_args ac_trap_action].
    rewrite lookup_map_values, HCk. cbn [option_map].
    split; [ reflexivity | split; [ reflexivity | exact Hrel ] ].
Qed.

Lemma with_kinds_length (params : list variable) :
  ∀ kinds : list flambda_kind, length (with_kinds params kinds) = length params.
Proof.
  induction params as [ | p params IH]; intros [ | k kinds]; simpl;
    rewrite ?IH; reflexivity.
Qed.

Lemma hm_with_params_kinds (metadata : handler_metadata)
    (kinds : list flambda_kind) :
  hm_params_kinds (with_params_kinds metadata kinds) = kinds.
Proof.
  unfold with_params_kinds. destruct (hm_is_exn_handler metadata); reflexivity.
Qed.

(* The parameters of a handler as the two rebuilds make them. *)
Lemma params_rel_handler (bound : list (variable * flambda_kind)) :
  ∀ mask : list bool,
    length mask = length bound ->
    forallb (fun '((p, kind), keep) => param_ok unboxed p kind keep)
      (combine bound mask)
    = true ->
    params_rel unboxed (flat_map shape_of_param (combine (map fst bound) mask))
      (filter_kept mask bound)
      (params_of_decisions bound
         (map
            (fun (p : (variable * flambda_kind) * bool) =>
               let '((x, kind), keep) := p in
               match fields_of unboxed x with
               | Some fields => Unbox fields
               | None => Decided (if keep then Keep kind else Delete)
               end)
            (combine bound mask)))
    ∧ Forall2 decision_match (combine (map fst bound) mask)
        (map
           (fun (p : (variable * flambda_kind) * bool) =>
              let '((x, kind), keep) := p in
              match fields_of unboxed x with
              | Some fields => Unbox fields
              | None => Decided (if keep then Keep kind else Delete)
              end)
           (combine bound mask)).
Proof.
  induction bound as [ | [x kind] bound IH]; intros [ | keep mask] Hlen Hok;
    simpl in Hlen; try discriminate Hlen.
  - split; constructor.
  - simpl in Hok. apply andb_true_iff in Hok as [Hx Hok].
    destruct (IH mask ltac:(lia) Hok) as [Hrel Hmatch].
    unfold param_ok in Hx. apply andb_true_iff in Hx as [Hfresh Hx].
    apply negb_true_iff in Hfresh.
    unfold FlambdaReaperUnboxingRefinesSpec.param_ok in Hx.
    cbn [map fst combine flat_map shape_of_param].
    destruct (fields_of unboxed x) as [fields | ] eqn:Hf.
    + apply andb_true_iff in Hx as [Hx Hkeep].
      apply andb_true_iff in Hx as [Hkind Hflat].
      apply kind_eqb_eq in Hkind. subst kind.
      split.
      * destruct keep.
        -- cbn [app filter_kept params_of_decisions decision_params].
           apply params_rel_unboxed; assumption.
        -- destruct fields; [ | discriminate Hkeep ].
           cbn [app filter_kept params_of_decisions decision_params
                fields_leaves unboxed_field_leaves].
           exact Hrel.
      * constructor; [ | exact Hmatch ]. unfold decision_match. rewrite Hf.
        split; [ reflexivity | ].
        destruct keep; [ left; reflexivity | right ].
        destruct fields; [ reflexivity | discriminate Hkeep ].
    + split.
      * destruct keep;
          cbn [app filter_kept params_of_decisions decision_params];
          [ | exact Hrel ].
        apply params_rel_plain; [ | exact Hrel ].
        apply plain_variable_iff. split; [ | exact Hfresh ].
        unfold Keyed.is_bound. unfold fields_of in Hf. rewrite Hf.
        reflexivity.
      * constructor; [ | exact Hmatch ]. unfold decision_match. rewrite Hf.
        destruct keep; [ right; split; [ eexists; reflexivity | reflexivity ]
                       | left; split; reflexivity ].
Qed.

Lemma handler_match (kinds : finite_map flambda_kind)
    (C : finite_map (list (variable * bool))) (is_exn : bool)
    (h : continuation_handler) :
  local_handler unboxed sol kinds C (handler_mask sol is_exn h) h = true ->
  Forall2 decision_match
    (combine (handler_variables h) (handler_mask sol is_exn h))
    (handler_decisions sol unboxed is_exn h)
  ∧ length (handler_mask sol is_exn h) = length (handler_variables h).
Proof.
  destruct h as [params metadata handler]. cbn [local_handler]. intros Hloc.
  apply andb_true_iff in Hloc as [Hloc _].
  apply andb_true_iff in Hloc as [Hlen Hparams]. apply Nat.eqb_eq in Hlen.
  pose proof (with_kinds_combine params (hm_params_kinds metadata) Hlen)
    as Hbound.
  set (bound := with_kinds params (hm_params_kinds metadata)) in *.
  set (mask := handler_mask sol is_exn (Handler params metadata handler))
    in *.
  assert (Hmlen : length mask = length bound)
    by (unfold mask, handler_mask; apply params_to_keep_length).
  assert (Hfst : map fst bound = params)
    by (rewrite Hbound; apply map_fst_combine, Hlen).
  rewrite <- Hbound in Hparams.
  destruct (params_rel_handler bound mask Hmlen Hparams) as [_ Hmatch].
  rewrite Hfst in Hmatch. cbn [handler_variables].
  split; [ exact Hmatch | ].
  rewrite Hmlen. unfold bound. apply with_kinds_length.
Qed.

Lemma conts_match_rec (kinds : finite_map flambda_kind)
    (C0 : finite_map (list (variable * bool)))
    (hs : list (continuation * continuation_handler)) :
  (∀ (k : continuation) (h : continuation_handler),
     (k, h) ∈ hs ->
     local_handler unboxed sol kinds C0 (handler_mask sol false h) h = true) ->
  conts_match
    (map (fun '(k, h) =>
            (k, combine (handler_variables h) (handler_mask sol false h)))
       hs)
    (map (fun '(k, h) => (k, handler_decisions sol unboxed false h)) hs).
Proof.
  induction hs as [ | [k h] hs IHhs]; intros Hhs; [ constructor | ].
  cbn [map]. constructor.
  - split; [ reflexivity | ].
    exact (proj1 (handler_match _ _ false h (Hhs k h (or_introl eq_refl)))).
  - apply IHhs. intros k' h' Hin. apply (Hhs k' h'). right. exact Hin.
Qed.

Lemma pairs_flat (fields fields_y : unboxed_fields) :
  forallb
    (fun '(f, u) =>
       match u, Keyed.lookup field_eqb f fields_y with
       | Not_unboxed _, Some (Not_unboxed _) => true
       | _, _ => false
       end)
    fields
  = true ->
  ∃ pairs : list (variable * variable * flambda_kind),
    unboxed_field_pairs Value (Unboxed fields) (Unboxed fields_y)
    = Some pairs
    ∧ Forall2
        (fun (fu : field * unboxed_field)
             (vvk : variable * variable * flambda_kind) =>
           let '(v, v', kind) := vvk in
           snd fu = Not_unboxed v ∧ kind = field_kind (fst fu)
           ∧ (fst fu, Not_unboxed v') ∈ fields_y)
        fields pairs.
Proof.
  induction fields as [ | [f u] fields IH]; intros Hf; simpl in Hf.
  - exists []. split; [ reflexivity | constructor ].
  - apply andb_true_iff in Hf as [Hfu Hf].
    destruct (IH Hf) as (pairs & Hp & Hall).
    destruct u as [v0 | nested]; [ | discriminate Hfu ].
    destruct (Keyed.lookup field_eqb f fields_y) as [[v | ] | ] eqn:Hl;
      try discriminate Hfu.
    exists ((v0, v, field_kind f) :: pairs). split.
    + change (unboxed_field_pairs Value
                (Unboxed ((f, Not_unboxed v0) :: fields))
                (Unboxed fields_y))
        with (match Keyed.lookup field_eqb f fields_y with
              | Some u' =>
                match unboxed_field_pairs (field_kind f) (Not_unboxed v0) u',
                      unboxed_field_pairs Value (Unboxed fields)
                        (Unboxed fields_y)
                with
                | Some p, Some q => Some (p ++ q)
                | _, _ => None
                end
              | None => None
              end).
      rewrite Hl, Hp. reflexivity.
    + constructor; [ | exact Hall ].
      split; [ reflexivity | split; [ reflexivity | ] ].
      exact (keyed_lookup_in field_eqb field_eqb_decides f _ fields_y Hl).
Qed.

Lemma lets_of_pairs (pairs : list (variable * variable * flambda_kind)) :
  ∀ body : expr,
    List.fold_left
      (fun body '(x, x', kind) => Let x kind (Simple (Var x')) body)
      pairs body
    = lets (map (fun '(x, x', kind) => (x, kind, Var x')) pairs) body.
Proof.
  induction pairs as [ | [[x x'] kind] pairs IH]; intros body;
    [ reflexivity | ].
  simpl. rewrite IH. reflexivity.
Qed.

(* The fields of an alias of an unboxed variable. *)
Lemma bind_fields_lets (fields fields_y : unboxed_fields) (body : expr) :
  forallb
    (fun '(f, u) =>
       match u, Keyed.lookup field_eqb f fields_y with
       | Not_unboxed _, Some (Not_unboxed _) => true
       | _, _ => false
       end)
    fields
  = true ->
  ∃ l : list (variable * flambda_kind * simple),
    bind_fields (Unboxed fields) (Unboxed fields_y) body = Some (lets l body)
    ∧ alias_entries fields_y fields l.
Proof.
  intros Hf. destruct (pairs_flat fields fields_y Hf) as (pairs & Hp & Hall).
  exists (map (fun '(x, x', kind) => (x, kind, Var x')) pairs).
  split; [ unfold bind_fields; rewrite Hp, lets_of_pairs; reflexivity | ].
  unfold alias_entries. clear -Hall.
  induction Hall as [ | [f u] [[v v'] kind] fields pairs (Hu & Hk & Hin) _ IH];
    [ constructor | ].
  cbn [map]. constructor; [ | exact IH ].
  simpl in Hu, Hk, Hin. split; [ exact Hu | split; [ exact Hk | ] ].
  exists v'. split; [ reflexivity | exact Hin ].
Qed.

Lemma lets_app (l l' : list (variable * flambda_kind * simple))
    (body : expr) :
  lets (l ++ l') body = lets l' (lets l body).
Proof. unfold lets. apply fold_left_app. Qed.

(* The fields of a block that is unboxed, some unboxed in turn. *)
Lemma bind_block_fields_lets2 (en : rebuild_env) (x : variable) (tag : Z)
    (args : list simple) (fields : unboxed_fields) :
  (∀ (f : field) (u : unboxed_field),
     (f, u) ∈ fields ->
     ∃ (i : nat) (a : simple),
       f = Field_block i Value ∧ nth_error args i = Some a
       ∧ s_field_used sol (Node_variable x) f = true
       ∧ kept_simple en a = true ∧ simple_kind en a = Value
       ∧ (((∃ v : variable, u = Not_unboxed v)
           ∧ plain_simple unboxed a = true)
          ∨ ∃ (nested : unboxed_fields) (b : variable)
              (fields_b : unboxed_fields),
              u = Unboxed nested ∧ a = Var b ∧ x <> b
              ∧ fields_of unboxed b = Some fields_b
              ∧ fields_alike nested fields_b = true)) ->
  ∀ body : expr,
    ∃ ls : list (list (variable * flambda_kind * simple)),
      bind_block_fields unboxed en tag args fields body
      = Some (lets (concat ls) body)
      ∧ Forall2
          (alloc_entry unboxed x
             (Prim (Variadic (Make_block (Block_kind_values tag)))
                (rebuild_fields sol en (Node_variable x) 0 args)))
          fields ls.
Proof.
  intros Hfields. unfold bind_block_fields.
  induction fields as [ | [f u] fields IH]; intros body.
  - exists []. split; [ reflexivity | constructor ].
  - destruct (Hfields f u (or_introl eq_refl))
      as (i & a & -> & Hnth & Hused & Hkept & Hkind & Hcase).
    assert (Hfield : nth_error (rebuild_fields sol en (Node_variable x) 0 args)
                       i
                     = Some a).
    { rewrite (rebuild_fields_nth en _ args 0 i a Hnth). simpl.
      unfold field_used. rewrite Hused, (kept_rewrite en a Hkept).
      reflexivity. }
    destruct Hcase as [((v & ->) & Ha)
                      | (nested & b & fb & -> & -> & Hxb & Hb & Halike)].
    + destruct (IH (fun f' u' Hin => Hfields f' u' (or_intror Hin))
                  (Let v Value (Simple a) body)) as (ls & Hl & Hentries).
      exists ([(v, Value, a)] :: ls). split.
      * simpl List.fold_left. rewrite Hnth, Hkind, kind_eqb_refl.
        rewrite (plain_not_bound a Ha). rewrite Hl.
        cbn [concat]. rewrite lets_app. reflexivity.
      * constructor; [ | exact Hentries ].
        cbn [alloc_entry]. exists Value, a.
        split; [ reflexivity | split; [ | exact Ha ] ].
        exists i.
        split; [ reflexivity | split; [ reflexivity | exact Hfield ] ].
    + destruct (bind_fields_lets nested fb body Halike) as (l & Hl & Hal).
      destruct (IH (fun f' u' Hin => Hfields f' u' (or_intror Hin))
                  (lets l body)) as (ls & Hls & Hentries).
      exists (l :: ls). split.
      * simpl List.fold_left. rewrite Hnth, Hkind, kind_eqb_refl.
        cbn [unboxed_of].
        change (unboxed_of_node unboxed (Node_variable b))
          with (fields_of unboxed b).
        rewrite Hb, Hl, Hls. cbn [concat]. rewrite lets_app. reflexivity.
      * constructor; [ | exact Hentries ].
        cbn [alloc_entry]. exists b, fb.
        split; [ | split; [ exact Hb | split; [ exact Hxb | exact Hal ] ] ].
        exists i.
        split; [ reflexivity | split; [ reflexivity | exact Hfield ] ].
Qed.

Definition ub_result (K : finite_map (list (option unboxed_fields)))
    (p q : expr) : Prop :=
  ub unboxed K p q
  ∧ free_continuations_of_expr p = free_continuations_of_expr q
  ∧ binds_statics p = binds_statics q.

Lemma ub_result_invalid (K : finite_map (list (option unboxed_fields))) :
  ub_result K Invalid Invalid.
Proof. split; [ constructor | split; reflexivity ]. Qed.

Lemma ac_rel_conts (K : finite_map (list (option unboxed_fields)))
    (ac ac' : apply_cont) :
  ac_rel unboxed K ac ac' ->
  apply_cont_continuations ac' = apply_cont_continuations ac.
Proof.
  intros (Hk & Htrap & _). unfold apply_cont_continuations.
  rewrite Hk, Htrap. reflexivity.
Qed.

Lemma pen_bind (en : unbox_env) (C : finite_map (list (variable * bool)))
    (bound : list (variable * flambda_kind)) :
  pen_of (ubind_variables en bound) C = bind_variables (pen_of en C) bound.
Proof. reflexivity. Qed.

Lemma rebuild_let_u_plain (en : rebuild_env) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  named_plain unboxed d = true ->
  rebuild_let_u sol unboxed [] en x kind d body
  = Let x kind (rebuild_named sol en x d) body.
Proof.
  intros Hd. unfold rebuild_let_u.
  destruct (field_read d) as [[f arg] | ] eqn:Hread.
  - assert (Harg : plain_simple unboxed arg = true).
    { destruct d as [s0 | p args | | | | | ]; simpl in Hd; try discriminate Hd;
        [ discriminate Hread | ].
      destruct p as [p | p | p | p]; simpl in Hread; try discriminate Hread.
      destruct p; simpl in Hread; try discriminate Hread;
        destruct args as [ | a [ | ]]; simpl in Hread; try discriminate Hread;
        try (destruct variant_only; simpl in Hread; try discriminate Hread);
        injection Hread as _ <-; simpl in Hd; rewrite andb_true_r in Hd;
        exact Hd. }
    rewrite (plain_not_bound arg Harg).
    assert (Hch : changed_of [] arg = None) by (destruct arg; reflexivity).
    rewrite Hch, (named_plain_u en x d Hd). reflexivity.
  - rewrite (named_plain_u en x d Hd). reflexivity.
Qed.

Lemma plain_rebuild_let_static (en : rebuild_env)
    (bindings : list static_binding) (body : expr) :
  plain_rebuild en (Let_static bindings body)
  = Let_static
      (flat_map
         (rebuild_binding F sol preserve_direct_calls unit_code_ids
            first_fresh en)
         bindings)
      (plain_rebuild (bind_variables en []) body).
Proof. reflexivity. Qed.

Lemma unbox_rebuild_let_static (en : unbox_env)
    (bindings : list static_binding) (body : expr) :
  unbox_rebuild en (Let_static bindings body)
  = Let_static
      (flat_map
         (rebuild_binding_u F sol cannot_change code_deps
            preserve_direct_calls unit_code_ids codes unboxed first_fresh_u []
            en)
         bindings)
      (unbox_rebuild (ubind_variables en []) body).
Proof. reflexivity. Qed.

Lemma plain_rebuild_let_cont (en : rebuild_env) (k : continuation)
    (h : continuation_handler) (body : expr) :
  plain_rebuild en (Let_cont_non_recursive k h body)
  = let mask := handler_mask sol (handler_is_exn_handler h) h in
    let body := plain_rebuild (bind_continuations en [(k, mask)]) body in
    if among k (free_continuations_of_expr body)
    then
      Let_cont_non_recursive k
        (rebuild_handler F sol preserve_direct_calls unit_code_ids
           first_fresh en true mask h)
        body
    else body.
Proof. reflexivity. Qed.

Lemma unbox_rebuild_let_cont (en : unbox_env) (k : continuation)
    (h : continuation_handler) (body : expr) :
  unbox_rebuild en (Let_cont_non_recursive k h body)
  = let decisions :=
      handler_decisions sol unboxed (handler_is_exn_handler h) h in
    let body :=
      unbox_rebuild (ubind_continuations en [(k, decisions)]) body in
    if among k (free_continuations_of_expr body)
    then
      Let_cont_non_recursive k
        (rebuild_handler_u F sol cannot_change code_deps
           preserve_direct_calls unit_code_ids codes unboxed first_fresh_u []
           en true decisions h)
        body
    else body.
Proof. reflexivity. Qed.

Lemma plain_rebuild_let_cont_rec (en : rebuild_env)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  plain_rebuild en (Let_cont_recursive handlers body)
  = let en :=
      bind_continuations en
        (map (fun '(k, h) => (k, handler_mask sol false h)) handlers)
    in
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k,
             rebuild_handler F sol preserve_direct_calls unit_code_ids
               first_fresh en false (handler_mask sol false h) h))
         handlers)
      (plain_rebuild en body).
Proof. reflexivity. Qed.

Lemma unbox_rebuild_let_cont_rec (en : unbox_env)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  unbox_rebuild en (Let_cont_recursive handlers body)
  = let en :=
      ubind_continuations en
        (map (fun '(k, h) => (k, handler_decisions sol unboxed false h))
           handlers)
    in
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k,
             rebuild_handler_u F sol cannot_change code_deps
               preserve_direct_calls unit_code_ids codes unboxed
               first_fresh_u [] en false
               (handler_decisions sol unboxed false h) h))
         handlers)
      (unbox_rebuild en body).
Proof. reflexivity. Qed.

Lemma plain_rebuild_handler (en : rebuild_env) (dead : bool)
    (mask : list bool) (params : list variable) (metadata : handler_metadata)
    (handler : expr) :
  rebuild_handler F sol preserve_direct_calls unit_code_ids first_fresh en
    dead mask (Handler params metadata handler)
  = let bound := with_kinds params (hm_params_kinds metadata) in
    Handler (filter_kept mask params)
      (with_params_kinds metadata (filter_kept mask (hm_params_kinds metadata)))
      (if dead && existsb (fun '(x, kind) => is_dead_var sol x kind) bound
       then Invalid
       else plain_rebuild (bind_variables en bound) handler).
Proof. reflexivity. Qed.

Lemma unbox_rebuild_handler (en : unbox_env) (dead : bool)
    (decisions : list decision) (params : list variable)
    (metadata : handler_metadata) (handler : expr) :
  rebuild_handler_u F sol cannot_change code_deps preserve_direct_calls
    unit_code_ids codes unboxed first_fresh_u [] en dead decisions
    (Handler params metadata handler)
  = let bound := with_kinds params (hm_params_kinds metadata) in
    let params := params_of_decisions bound decisions in
    Handler (map fst params) (with_params_kinds metadata (map snd params))
      (if dead && existsb (fun '(x, kind) => is_dead_var sol x kind) bound
       then Invalid
       else unbox_rebuild (ubind_variables en bound) handler).
Proof. reflexivity. Qed.

Lemma every_in {A : Type} (P : A -> Prop) (l : list A) (a : A) :
  every P l -> a ∈ l -> P a.
Proof.
  induction l as [ | b l IH]; intros Hl Ha; [ destruct Ha | ].
  destruct Hl as [Hb Hl]. destruct Ha as [<- | Ha];
    [ exact Hb | exact (IH Hl Ha) ].
Qed.

Lemma local_bindings_rebuilt (en : unbox_env)
    (C : finite_map (list (variable * bool)))
    (bindings : list static_binding) :
  forallb (local_binding unboxed) bindings = true ->
  flat_map
    (rebuild_binding_u F sol cannot_change code_deps preserve_direct_calls
       unit_code_ids codes unboxed first_fresh_u [] en)
    bindings
  = flat_map
      (rebuild_binding F sol preserve_direct_calls unit_code_ids first_fresh
         (pen_of en C))
      bindings
  ∧ forallb (local_binding unboxed)
      (flat_map
         (rebuild_binding F sol preserve_direct_calls unit_code_ids
            first_fresh (pen_of en C))
         bindings)
    = true.
Proof.
  induction bindings as [ | binding bindings IH]; simpl;
    [ split; reflexivity | ].
  rewrite andb_true_iff. intros [Hb Hbs]. destruct (IH Hbs) as [Heq Hplain].
  destruct binding as [id c | id | sym [ | | tag fields | | | ]];
    simpl in Hb; try discriminate Hb.
  simpl. rewrite Heq.
  destruct (is_symbol_used sol sym);
    simpl; [ | split; [ reflexivity | exact Hplain ] ].
  rewrite (rebuild_fields_u (ue_env en) (Node_symbol sym) fields Hb 0).
  rewrite (rebuild_fields_kinds (pen_of en C) (ue_env en) (Node_symbol sym)
             fields eq_refl 0).
  split; [ reflexivity | ]. rewrite Hplain, andb_true_r.
  apply rebuild_fields_plain, Hb.
Qed.

Lemma rebuild_ub :
  (∀ e : expr,
     ∀ (en : unbox_env) (C : finite_map (list (variable * bool)))
       (ce : finite_map (list node)),
       uinv en -> conts_match C (ue_decisions en) ->
       local_expr unboxed sol (env_kinds (ue_env en)) C e = true ->
       edges ce e ->
       ub_result (K_of C) (plain_rebuild (pen_of en C) e)
         (unbox_rebuild en e))
  ∧ (∀ h : continuation_handler,
       ∀ (en : unbox_env) (C : finite_map (list (variable * bool)))
         (ce : finite_map (list node)) (is_exn : bool),
         uinv en -> conts_match C (ue_decisions en) ->
         local_handler unboxed sol (env_kinds (ue_env en)) C
           (handler_mask sol is_exn h) h
         = true ->
         FlambdaReaperFlowProofs.handler_edges (graph_of_facts F) noted deps
           cds kept_slots codes0 symbols True None ce h ->
         ∀ dead : bool,
           ∃ (ps ps' : list (variable * flambda_kind))
             (metadata metadata' : handler_metadata) (hp hu : expr),
             rebuild_handler F sol preserve_direct_calls unit_code_ids
               first_fresh (pen_of en C) dead (handler_mask sol is_exn h) h
             = Handler (map fst ps) metadata hp
             ∧ rebuild_handler_u F sol cannot_change code_deps
                 preserve_direct_calls unit_code_ids codes unboxed
                 first_fresh_u [] en dead
                 (handler_decisions sol unboxed is_exn h) h
               = Handler (map fst ps') metadata' hu
             ∧ params_rel unboxed
                 (flat_map shape_of_param
                    (combine (handler_variables h)
                       (handler_mask sol is_exn h)))
                 ps ps'
             ∧ hm_params_kinds metadata = map snd ps
             ∧ hm_params_kinds metadata' = map snd ps'
             ∧ Forall2 decision_match
                 (combine (handler_variables h) (handler_mask sol is_exn h))
                 (handler_decisions sol unboxed is_exn h)
             ∧ length (handler_mask sol is_exn h)
               = length (handler_variables h)
             ∧ ub_result (K_of C) hp hu).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - (* Let *)
    intros x kind d body IH en C ce Hen HC Hloc Hedges. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hloc Hbody].
    apply andb_true_iff in Hloc as [Hxf Hbind]. apply negb_true_iff in Hxf.
    destruct Hedges as [Hnamed_edges Hbody_edges].
    specialize (IH (ubind_variables en [(x, kind)]) C ce
                  (uinv_bind en [(x, kind)]) HC Hbody Hbody_edges).
    rewrite pen_bind in IH.
    unfold plain_rebuild, unbox_rebuild in *. cbn [rebuild_expr rebuild_expr_u].
    unfold is_pending_projection.
    rewrite (uinv_projections en Hen), pen_projections.
    rewrite (rebuild_named_kinds (pen_of en C) (ue_env en) x d eq_refl).
    assert (Hd : (∃ s, d = Simple s) ∨ (∃ p args, d = Prim p args)).
    { destruct (fields_of unboxed x) as [fields | ] eqn:Hx.
      - destruct (unboxed_binding_cases _ x kind d fields Hbind)
          as (_ & [(tag & args & -> & _) | [(bn & c & -> & _)
                  | [(y & fy & -> & _)
                    | (bk & mut & i & y & fy & n & -> & _)]]]);
          [ right | right | left | right ]; eauto.
      - destruct (plain_binding_cases x kind d Hbind)
          as [Hp | (p & y & f & fs & -> & _)]; [ | right; eauto ].
        destruct d; try discriminate Hp; eauto. }
    destruct IH as (IHub & IHfc & IHbs).
    destruct Hd as [(s0 & ->) | (p & args & ->)]; cbn beta iota;
      (destruct (is_dead_var sol x kind); [ exact (ub_result_invalid _) | ]);
      (destruct (is_var_used sol x kind) eqn:Hused;
       [ | split; [ exact IHub | split; assumption ] ]);
      change (unboxed_of_node unboxed (Node_variable x))
        with (fields_of unboxed x);
      change (changed_of_node [] (Node_variable x)) with (@None closure_repr);
      cbn iota;
      (destruct (fields_of unboxed x) as [fields | ] eqn:Hx).
    + (* An alias of an unboxed variable. *)
      destruct (unboxed_binding_cases _ x kind _ fields Hbind)
        as (-> & [(tag & args & Heq & _) | [(bn & c & Heq & _)
                 | [(y & fields_y & Heq & Hne & Hy & Hflat & Hyused)
                   | (bk & mut & i & y & fy & n & Heq & _)]]]);
        try discriminate Heq.
      injection Heq as ->.
      cbn [rebuild_named rewrite_simple].
      unfold kind_of_variable in Hyused |- *.
      cbn [env_kinds kinds_env] in Hyused. rewrite Hyused.
      unfold rebuild_unboxed_binding.
      destruct (bind_fields_lets fields fields_y
                  (rebuild_expr_u F sol cannot_change code_deps
                     preserve_direct_calls unit_code_ids codes unboxed
                     first_fresh_u [] (ubind_variables en [(x, Value)]) body)
                  Hflat)
        as (l & Hl & Hentries).
      cbn [unboxed_of].
      change (unboxed_of_node unboxed (Node_variable y))
        with (fields_of unboxed y).
      rewrite Hy, Hl.
      destruct (lets_free_continuations l
                  (rebuild_expr_u F sol cannot_change code_deps
                     preserve_direct_calls unit_code_ids codes unboxed
                     first_fresh_u [] (ubind_variables en [(x, Value)]) body))
        as [Hfc Hbs].
      split; [ exact (ub_alias unboxed _ x y fields fields_y l _ _ Hx Hxf Hy
                        Hne Hentries IHub)
             | ].
      cbn [free_continuations_of_expr binds_statics]. rewrite Hfc, Hbs.
      split; assumption.
    + (* A simple. *)
      destruct (plain_binding_cases x kind _ Hbind)
        as [Hp | (p & y & f & fs & Heq & _)]; [ | discriminate Heq ].
      rewrite (rebuild_let_u_plain _ x kind _ _ Hp).
      split; [ | split; assumption ].
      constructor; [ | apply named_plain_rebuilt, Hp | exact IHub ].
      apply plain_variable_iff. unfold Keyed.is_bound.
      unfold fields_of in Hx. rewrite Hx. split; [ reflexivity | exact Hxf ].
    + (* An unboxed allocation, or the load of a field unboxed in turn. *)
      destruct (unboxed_binding_cases _ x kind _ fields Hbind)
        as (-> & [(tag & args' & Heq & Hfields)
                 | [(bn & c & Heq & Hc & Hfields)
                   | [(y & fy & Heq & _)
                     | (bk & mut & i & y & fy & nested & Heq & Hne & Hy & Hny
                        & Halike)]]]);
        [ | | discriminate Heq | ];
        injection Heq as -> ->; simpl in Hnamed_edges.
      * destruct (bind_block_fields_lets2 (ue_env en) x tag args' fields)
          with (body := rebuild_expr_u F sol cannot_change code_deps
                          preserve_direct_calls unit_code_ids codes unboxed
                          first_fresh_u [] (ubind_variables en [(x, Value)])
                          body)
          as (ls & Hl & Hentries).
        { intros f u Hin.
          destruct (Hfields f u Hin) as (i & a & -> & Hnth & Hkind & Hcase).
          rewrite kinds_env_kind in Hkind.
          destruct (field_kept x fields _ u a (ue_env en) Hx Hin
                      (proj1 Hnamed_edges i a Hnth) eq_refl)
            as [Hused' Hkept].
          exists i, a. repeat split; assumption. }
        unfold rebuild_unboxed_binding. rewrite Hl. cbn [rebuild_named].
        destruct (lets_free_continuations (concat ls)
                    (rebuild_expr_u F sol cannot_change code_deps
                       preserve_direct_calls unit_code_ids codes unboxed
                       first_fresh_u [] (ubind_variables en [(x, Value)])
                       body))
          as [Hfc Hbs].
        split; [ exact (ub_alloc unboxed _ x _ fields ls _ _ Hx Hxf Hentries
                          IHub)
               | ].
        cbn [free_continuations_of_expr binds_statics]. rewrite Hfc, Hbs.
        split; assumption.
      * destruct (bind_boxed_number_lets bn c fields
                    (rebuild_named sol (ue_env en) x
                       (Prim (Unary (Box_number bn)) [c])) Hc)
          with (body := rebuild_expr_u F sol cannot_change code_deps
                          preserve_direct_calls unit_code_ids codes unboxed
                          first_fresh_u [] (ubind_variables en [(x, Value)])
                          body)
          as (l & Hl & Hentries).
        { intros Hne. destruct fields as [ | [f u] fields]; [ easy | ].
          destruct (Hfields f u (or_introl eq_refl)) as (v & -> & ->).
          destruct (field_kept x _ _ (Not_unboxed v) c (ue_env en) Hx
                      (or_introl eq_refl) (proj1 Hnamed_edges) eq_refl)
            as [Hused' Hkept].
          cbn [rebuild_named].
          unfold field_used. rewrite Hused'. simpl.
          rewrite (kept_rewrite _ c Hkept). reflexivity. }
        { intros f u Hin. exact (Hfields f u Hin). }
        unfold rebuild_unboxed_binding. rewrite Hl.
        destruct (lets_free_continuations l
                    (rebuild_expr_u F sol cannot_change code_deps
                       preserve_direct_calls unit_code_ids codes unboxed
                       first_fresh_u [] (ubind_variables en [(x, Value)])
                       body))
          as [Hfc Hbs].
        destruct (alloc_entry_flat unboxed x _ fields l Hentries)
          as [Hentries2 Hconcat].
        split; [ rewrite <- Hconcat at 1;
                 exact (ub_alloc unboxed _ x _ fields _ _ _ Hx Hxf Hentries2
                          IHub)
               | ].
        cbn [free_continuations_of_expr binds_statics]. rewrite Hfc, Hbs.
        split; assumption.
      * (* The load of a field unboxed in turn. *)
        cbn [rebuild_named]. unfold rebuild_unboxed_binding, load_unboxed_field.
        cbn [unboxed_of].
        change (unboxed_of_node unboxed (Node_variable y))
          with (fields_of unboxed y).
        rewrite Hy, Hny.
        destruct (bind_fields_lets fields nested
                    (rebuild_expr_u F sol cannot_change code_deps
                       preserve_direct_calls unit_code_ids codes unboxed
                       first_fresh_u [] (ubind_variables en [(x, Value)])
                       body)
                    Halike)
          as (l & Hl & Hentries).
        rewrite Hl.
        destruct (lets_free_continuations l
                    (rebuild_expr_u F sol cannot_change code_deps
                       preserve_direct_calls unit_code_ids codes unboxed
                       first_fresh_u [] (ubind_variables en [(x, Value)])
                       body))
          as [Hfc Hbs].
        split.
        -- apply (ub_load unboxed _ x Value _ (Field_block (Z.to_nat i) Value)
                    (rewrite_simple sol (ue_env en) (Var y)) y fields fy
                    nested l); try assumption.
           ++ reflexivity.
           ++ simpl. destruct (is_var_used sol y _); [ left; reflexivity | ].
              right. eexists. reflexivity.
           ++ exact (keyed_lookup_in field_eqb field_eqb_decides _ _ fy Hny).
        -- cbn [free_continuations_of_expr binds_statics]. rewrite Hfc, Hbs.
           split; assumption.
    + (* A primitive of a variable that is not unboxed. *)
      destruct (plain_binding_cases x kind _ Hbind)
        as [Hp | (p' & y & f & fs & Heq & Hread & Hreads & Hnamed & Hy
                  & Hfield)].
      * rewrite (rebuild_let_u_plain _ x kind _ _ Hp).
        split; [ | split; assumption ].
        constructor; [ | apply named_plain_rebuilt, Hp | exact IHub ].
        apply plain_variable_iff. unfold Keyed.is_bound.
        unfold fields_of in Hx. rewrite Hx. split; [ reflexivity | exact Hxf ].
      * rewrite Heq in Hnamed |- *. rewrite (Hnamed (ue_env en)).
        destruct (Hfield Hused) as (v & Hv).
        unfold rebuild_let_u.
        assert (Hfr : field_read (Prim (Unary p') [Var y]) = Some (f, Var y)).
        { clear -Hread. destruct p'; try discriminate Hread; exact Hread. }
        rewrite Hfr. cbn [unboxed_of]. change (unboxed_of_node unboxed
          (Node_variable y)) with (fields_of unboxed y).
        rewrite Hy, Hv.
        split; [ | split; assumption ].
        apply (ub_read unboxed _ x kind
                 (Prim (Unary p') [rewrite_simple sol (ue_env en) (Var y)]) f
                 (rewrite_simple sol (ue_env en) (Var y)) y v fs).
        -- apply plain_variable_iff. unfold Keyed.is_bound.
           unfold fields_of in Hx. rewrite Hx.
           split; [ reflexivity | exact Hxf ].
        -- apply Hreads.
        -- simpl. destruct (is_var_used sol y _); [ left; reflexivity | ].
           right. eexists. reflexivity.
        -- exact Hy.
        -- exact (keyed_lookup_in field_eqb field_eqb_decides f _ fs Hv).
        -- exact IHub.
  - (* Let_static *)
    intros bindings body _ IH en C ce Hen HC Hloc Hedges. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hbs Hbody].
    destruct (IH (ubind_variables en []) C ce (uinv_bind en []) HC Hbody
                (proj2 Hedges))
      as (IHub & IHfc & IHbs).
    rewrite pen_bind in IHub, IHfc, IHbs.
    destruct (local_bindings_rebuilt en C bindings Hbs) as [Heq Hplain].
    rewrite plain_rebuild_let_static, unbox_rebuild_let_static, Heq.
    split; [ constructor; [ exact Hplain | exact IHub ] | ].
    cbn [free_continuations_of_expr binds_statics]. rewrite IHfc, IHbs.
    split; reflexivity.
  - (* Let_cont_non_recursive *)
    intros k h body IHh IH en C ce Hen HC Hloc Hedges. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hh Hbody].
    destruct Hedges as [Hh_edges Hbody_edges].
    rewrite plain_rebuild_let_cont, unbox_rebuild_let_cont. cbv zeta.
    destruct (IHh en C ce (handler_is_exn_handler h) Hen HC Hh Hh_edges true)
      as (ps & ps' & md & md' & hp & hu & Hp & Hu & Hrel & Hmd & Hmd'
          & Hmatch & Hlen & Hhub & Hhfc & Hhbs).
    set (is_exn := handler_is_exn_handler h) in *.
    set (mask := handler_mask sol is_exn h) in *.
    set (ds := handler_decisions sol unboxed is_exn h) in *.
    set (C' := (k, combine (handler_variables h) mask) :: C).
    assert (HC' : conts_match C' (ue_decisions
                                   (ubind_continuations en [(k, ds)]))).
    { constructor; [ split; [ reflexivity | exact Hmatch ] | exact HC ]. }
    assert (Hpen : pen_of (ubind_continuations en [(k, ds)]) C'
                   = bind_continuations (pen_of en C) [(k, mask)]).
    { unfold pen_of, C'. cbn [map env_conts env_kinds ue_env
        ubind_continuations bind_continuations app].
      rewrite map_snd_combine by exact (eq_sym Hlen). reflexivity. }
    destruct (IH (ubind_continuations en [(k, ds)]) C' _
                (uinv_continuations en [(k, ds)]) HC' Hbody Hbody_edges)
      as (IHub & IHfc & IHbs).
    rewrite Hpen in IHub, IHfc, IHbs.
    rewrite <- IHfc.
    destruct (among k _) eqn:Hk.
    + rewrite Hp, Hu.
      split; [ eapply ub_let_cont; eassumption | ].
      cbn [free_continuations_of_expr free_continuations_of_handler
           binds_statics handler_binds_statics].
      rewrite Hhfc, IHfc, Hhbs, IHbs. split; reflexivity.
    + split; [ | split; assumption ].
      apply (ub_agree unboxed _ _ _ IHub). intros k' Hk'.
      unfold C', K_of. cbn [map]. rewrite lookup_cons.
      destruct (Pos.eqb k' k) eqn:E; [ | reflexivity ].
      apply Pos.eqb_eq in E. subst k'. apply among_In in Hk'.
      rewrite Hk' in Hk. discriminate.
  - (* Let_cont_recursive *)
    intros handlers body IHh IH en C ce Hen HC Hloc Hedges. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hhs Hbody].
    rewrite forallb_forall in Hhs.
    destruct Hedges as [Hhs_edges Hbody_edges].
    rewrite plain_rebuild_let_cont_rec, unbox_rebuild_let_cont_rec. cbv zeta.
    set (dss := map (fun '(k, h) => (k, handler_decisions sol unboxed false h))
                  handlers).
    set (masks := map (fun '(k, h) => (k, handler_mask sol false h)) handlers).
    set (CR := map (fun '(k, h) =>
                      (k, combine (handler_variables h)
                            (handler_mask sol false h)))
                 handlers).
    set (S := map (fun '(k, h) =>
                     (k, flat_map shape_of_param
                           (combine (handler_variables h)
                              (handler_mask sol false h))))
                handlers).
    set (C' := CR ++ C).
    assert (Hhs' : ∀ (k : continuation) (h : continuation_handler),
               (k, h) ∈ handlers ->
               local_handler unboxed sol (env_kinds (ue_env en)) C'
                 (handler_mask sol false h) h = true)
      by (intros k h Hin; exact (Hhs _ Hin)).
    clear Hhs.
    assert (HC' : conts_match C' (ue_decisions
                                   (ubind_continuations en dss))).
    { cbn [ue_decisions ubind_continuations]. apply Forall2_app; [ | exact HC ].
      exact (conts_match_rec _ C' handlers Hhs'). }
    assert (Hpen : pen_of (ubind_continuations en dss) C'
                   = bind_continuations (pen_of en C) masks).
    { unfold pen_of, C', bind_continuations.
      cbn [ue_env ubind_continuations env_conts env_kinds].
      rewrite map_app.
      match goal with
      | |- context [map ?f CR] => assert (Hm : map f CR = masks)
      end.
      { unfold CR, masks. rewrite map_map.
        apply map_ext_in. intros [k h] Hin. rewrite map_snd_combine;
          [ reflexivity | ].
        symmetry. exact (proj2 (handler_match _ _ false h (Hhs' k h Hin))). }
      f_equal; try reflexivity. rewrite <- Hm. reflexivity. }
    assert (HK : K_of C' = S ++ K_of C).
    { unfold C', K_of, CR, S. rewrite map_app, map_map. f_equal.
      apply map_ext. intros [k h]. reflexivity. }
    destruct (IH (ubind_continuations en dss) C' _
                (uinv_continuations en dss) HC' Hbody Hbody_edges)
      as (IHub & IHfc & IHbs).
    rewrite Hpen in IHub, IHfc, IHbs.
    assert (Hhandlers : ∀ hs : list (continuation * continuation_handler),
      (∀ (k : continuation) (h : continuation_handler),
         (k, h) ∈ hs -> (k, h) ∈ handlers) ->
      let hps :=
        map
          (fun '(k, h) =>
             (k,
              rebuild_handler F sol preserve_direct_calls unit_code_ids
                first_fresh (bind_continuations (pen_of en C) masks) false
                (handler_mask sol false h) h))
          hs
      in
      let hus :=
        map
          (fun '(k, h) =>
             (k,
              rebuild_handler_u F sol cannot_change code_deps
                preserve_direct_calls unit_code_ids codes unboxed
                first_fresh_u [] (ubind_continuations en dss) false
                (handler_decisions sol unboxed false h) h))
          hs
      in
      ub_handlers unboxed (K_of C') hps hus
        (map (fun '(k, h) =>
                (k, flat_map shape_of_param
                      (combine (handler_variables h)
                         (handler_mask sol false h))))
           hs)
      ∧ handlers_bind_statics hps = handlers_bind_statics hus
      ∧ flat_map (fun '(_, h) => free_continuations_of_handler h) hps
        = flat_map (fun '(_, h) => free_continuations_of_handler h) hus
      ∧ map fst hps = map fst hus).
    { induction hs as [ | [k h] hs IHhs]; intros Hsub; cbv zeta;
        [ cbn [map]; split; [ apply ub_handlers_nil | ];
          split; [ reflexivity | split; reflexivity ] | ].
      destruct IHhs as (Hub_hs & Hbs_hs & Hfc_hs & Hfst_hs).
      { intros k' h' Hin. apply Hsub. right. exact Hin. }
      assert (Hin : (k, h) ∈ handlers) by (apply Hsub; left; reflexivity).
      destruct (IHh k h Hin (ubind_continuations en dss) C' _ false
                  (uinv_continuations en dss) HC' (Hhs' _ _ Hin)
                  (every_in _ _ (k, h) Hhs_edges Hin) false)
        as (ps & ps' & md & md' & hp & hu & Hp & Hu & Hrel & Hmd & Hmd' & _
            & _ & Hhub & Hhfc & Hhbs).
      rewrite Hpen in Hp. cbv zeta. cbn [map]. rewrite Hp, Hu.
      split; [ apply ub_handlers_cons; assumption | ].
      unfold handlers_bind_statics in *. cbn [existsb flat_map map fst].
      cbn [free_continuations_of_handler handler_binds_statics].
      rewrite Hhbs, Hbs_hs, Hhfc, Hfc_hs, Hfst_hs. split; [ | split ];
        reflexivity. }
    destruct (Hhandlers handlers (fun k h Hin => Hin))
      as (Hub_hs & Hbs_hs & Hfc_hs & Hfst_hs).
    fold S in Hub_hs.
    rewrite HK in Hub_hs, IHub.
    split; [ apply (ub_let_cont_rec unboxed (K_of C) S); assumption | ].
    unfold handlers_bind_statics in Hbs_hs.
    cbn [free_continuations_of_expr binds_statics].
    rewrite Hfst_hs, Hfc_hs, IHfc, IHbs, Hbs_hs. split; reflexivity.
  - (* Apply *)
    intros ap en C ce Hen HC Hloc. discriminate Hloc.
  - (* Apply_cont *)
    intros ac en C ce Hen HC Hloc _. unfold plain_rebuild, unbox_rebuild.
    cbn [rebuild_expr rebuild_expr_u]. simpl in Hloc.
    unfold rebuild_apply_cont, rebuild_apply_cont_u.
    destruct (rewrite_apply_cont_rel en C ac HC Hloc)
      as [[Hp Hu] | (ac_p & ac_u & Hp & Hu & Hrel)];
      rewrite Hp, Hu; [ exact (ub_result_invalid _) | ].
    split; [ constructor; exact Hrel | ].
    cbn [free_continuations_of_expr binds_statics].
    rewrite (ac_rel_conts _ _ _ Hrel). split; reflexivity.
  - (* Switch *)
    intros scrutinee arms en C ce Hen HC Hloc _. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hs Harms].
    unfold plain_rebuild, unbox_rebuild. cbn [rebuild_expr rebuild_expr_u].
    unfold rebuild_switch, rebuild_switch_u.
    assert (Hcompact : ∀ (d : Z) (ac : apply_cont),
               (d, ac) ∈ Keyed.compact Z.eqb arms ->
               apply_cont_ok unboxed sol (env_kinds (ue_env en)) C ac = true).
    { intros d ac Hin.
      apply (keyed_in_compact Z.eqb z_eqb_decides_equality) in Hin.
      apply (keyed_lookup_in Z.eqb z_eqb_decides_equality) in Hin.
      rewrite forallb_forall in Harms. exact (Harms _ Hin). }
    assert (Harms' :
      ∀ l : list (Z * apply_cont),
        (∀ (d : Z) (ac : apply_cont), (d, ac) ∈ l ->
           apply_cont_ok unboxed sol (env_kinds (ue_env en)) C ac = true) ->
        Forall2
          (fun arm arm' : Z * apply_cont =>
             fst arm' = fst arm ∧ ac_rel unboxed (K_of C) (snd arm) (snd arm'))
          (flat_map
             (fun '(d, ac) =>
                match rewrite_apply_cont sol (pen_of en C) ac with
                | Some ac => [(d, ac)]
                | None => []
                end) l)
          (flat_map
             (fun '(d, ac) =>
                match rewrite_apply_cont_u sol unboxed en ac with
                | Some ac => [(d, ac)]
                | None => []
                end) l)).
    { induction l as [ | [d ac] l IHl]; intros Hl; [ constructor | ].
      assert (Hrest : Forall2
          (fun arm arm' : Z * apply_cont =>
             fst arm' = fst arm ∧ ac_rel unboxed (K_of C) (snd arm) (snd arm'))
          (flat_map
             (fun '(d, ac) =>
                match rewrite_apply_cont sol (pen_of en C) ac with
                | Some ac => [(d, ac)]
                | None => []
                end) l)
          (flat_map
             (fun '(d, ac) =>
                match rewrite_apply_cont_u sol unboxed en ac with
                | Some ac => [(d, ac)]
                | None => []
                end) l)).
      { apply IHl. intros d' ac' Hin. apply (Hl d' ac'). right. exact Hin. }
      cbn [flat_map].
      destruct (rewrite_apply_cont_rel en C ac HC (Hl d ac (or_introl eq_refl)))
        as [[Hp Hu] | (ac_p & ac_u & Hp & Hu & Hrel)];
        rewrite Hp, Hu; [ exact Hrest | ].
      constructor; [ split; [ reflexivity | exact Hrel ] | exact Hrest ]. }
    pose proof (Harms' _ Hcompact) as Hrel.
    rewrite (rewrite_simple_kinds (pen_of en C) (ue_env en) scrutinee eq_refl).
    rewrite (rewrite_simple_u _ scrutinee Hs).
    destruct Hrel as [ | arm arm' l l' Harm Hrel ];
      [ exact (ub_result_invalid _) | ].
    cbn [Keyed.is_empty].
    split; [ apply ub_switch; [ apply rewrite_simple_plain, Hs | ] | ].
    { constructor; assumption. }
    split; [ | reflexivity ].
    cbn [free_continuations_of_expr flat_map].
    destruct Harm as [_ Hac]. rewrite (ac_rel_conts _ _ _ Hac). f_equal.
    clear -Hrel. induction Hrel as [ | a a' l l' [_ Ha] _ IH ];
      [ reflexivity | ].
    cbn [flat_map]. rewrite (ac_rel_conts _ _ _ Ha), IH. reflexivity.
  - (* Invalid *)
    intros en C ce Hen HC Hloc _. exact (ub_result_invalid _).
  - (* Handler *)
    intros params metadata handler IH en C ce is_exn Hen HC Hloc Hedges dead.
    simpl in Hloc.
    apply andb_true_iff in Hloc as [Hloc Hbody].
    apply andb_true_iff in Hloc as [Hlen Hparams]. apply Nat.eqb_eq in Hlen.
    pose proof (with_kinds_combine params (hm_params_kinds metadata) Hlen)
      as Hbound.
    set (bound := with_kinds params (hm_params_kinds metadata)) in *.
    set (mask := handler_mask sol is_exn (Handler params metadata handler))
      in *.
    assert (Hmlen : length mask = length bound)
      by (unfold mask, handler_mask; apply params_to_keep_length).
    assert (Hfst : map fst bound = params)
      by (rewrite Hbound; apply map_fst_combine, Hlen).
    assert (Hsnd : map snd bound = hm_params_kinds metadata)
      by (rewrite Hbound; apply map_snd_combine, Hlen).
    rewrite <- Hbound in Hparams.
    destruct (params_rel_handler bound mask Hmlen Hparams) as [Hrel Hmatch].
    rewrite Hfst in Hrel, Hmatch.
    assert (Hbody' : local_expr unboxed sol
                       (env_kinds (ue_env (ubind_variables en bound))) C
                       handler = true).
    { cbn [ue_env ubind_variables ubind_variables_then ubind_set
           bind_variables_then env_kinds].
      rewrite Hbound. exact Hbody. }
    destruct (IH _ C ce (uinv_bind en bound) HC Hbody' Hedges)
      as (IHub & IHfc & IHbs).
    rewrite pen_bind in IHub, IHfc, IHbs.
    rewrite plain_rebuild_handler, unbox_rebuild_handler. cbv zeta.
    fold bound.
    exists (filter_kept mask bound),
      (params_of_decisions bound
         (handler_decisions sol unboxed is_exn
            (Handler params metadata handler))),
      (with_params_kinds metadata
         (filter_kept mask (hm_params_kinds metadata))),
      (with_params_kinds metadata
         (map snd
            (params_of_decisions bound
               (handler_decisions sol unboxed is_exn
                  (Handler params metadata handler))))).
    eexists _, _.
    split; [ rewrite <- filter_kept_map, Hfst; reflexivity | ].
    split; [ reflexivity | ].
    split; [ exact Hrel | ].
    split; [ rewrite hm_with_params_kinds, <- Hsnd, filter_kept_map;
             reflexivity | ].
    split; [ apply hm_with_params_kinds | ].
    split; [ exact Hmatch | ].
    split; [ rewrite Hmlen; unfold bound; apply with_kinds_length | ].
    destruct (dead && _); [ exact (ub_result_invalid _) | ].
    split; [ exact IHub | split; assumption ].
Qed.

End Rebuilds.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Lemma uinv_empty : uinv empty_unbox_env.
Proof. split; reflexivity. Qed.

Theorem rebuild_unbox_refines_rebuild :
  rebuild_unbox_refines_rebuild_statement.
Proof.
  intros flag is_local u e T cannot_change code_deps dominated to_unbox
    preserve_direct_calls Hslots Hfragment Hclosed Hlocal.
  set (F := traverse_unit flag u e) in *.
  unfold local_unboxing in Hlocal.
  pose proof (unit_unboxed_fields_fresh F T dominated to_unbox e)
    as [Hdistinct Hfresh].
  unfold rebuild_unit_unbox, rebuild_unit_unbox_repr.
  destruct (unit_unboxed_fields F T dominated to_unbox e) as [unboxed next]
    eqn:Hunboxed.
  cbn [fst] in Hlocal, Hdistinct, Hfresh.
  rename Hlocal into Hloc.
  pose proof (closure_free_reads_blocks flag u e
                (proj1 (local_closure_free unboxed (solution_of_tables T))
                   e [] [] Hloc))
    as Hreads.
  apply (proj1 (strong_local unboxed (solution_of_tables T)) e [] [] Hloc)
    in Hfresh.
  rename Hfresh into Hloc'.
  assert (Hchanged :
    unit_changed_representations (fun _ => false) F T dominated to_unbox [] e
    = []) by reflexivity.
  rewrite Hchanged.
  pose proof (check_closed_sound is_local F T Hclosed) as Hcl.
  destruct (FlambdaReaperFlowProofs.traverse_unit_facts flag u e True
              Hfragment (or_introl I))
    as (noted & deps & cds & Hedges & _).
  assert (Hfields_read :
    ∀ (x : variable) (fields : unboxed_fields) (f : field)
      (v : unboxed_field),
      fields_of unboxed x = Some fields -> (f, v) ∈ fields ->
      ∃ w base : node,
        g_accessor (graph_of_facts F) w f base
        ∧ s_usages (solution_of_tables T) (Node_variable x) base = true
        ∧ s_has_usage (solution_of_tables T) w = true).
  { intros x fields f v Hx Hin. apply (fields_of_in unboxed) in Hx.
    unfold unit_unboxed_fields in Hunboxed.
    assert (Hx' : (Node_variable x, fields)
                  ∈ fst (unboxed_fields_of_names F T (bound_code_my_closures e)
                           dominated to_unbox (fresh_name (names_of e))))
      by (rewrite Hunboxed; exact Hx).
    destruct (unboxed_fields_of_names_in F T _ dominated to_unbox _ _ _ Hx')
      as (a & next' & ->).
    destruct (mk_unboxed_fields_in F T _ dominated to_unbox _ a _ next' f _
                Hin)
      as (use & Huse).
    destruct (get_fields_in F T _ f use Huse)
      as (w & base & Hacc & Hbase & Hw).
    rewrite (add_usages_none F T _ Hreads) in Hbase.
    apply mem_node_In in Hbase.
    destruct (get_direct_usages_in T _ _ Hbase) as [Hus _].
    exists w, base. split; [ exact Hacc | split; [ exact Hus | ] ].
    destruct Hw as [Hw | Hw]; [ | exact Hw ].
    exact (has_usage_any _ _ _ Hcl _ Hw). }
  destruct (proj1 (rebuild_ub F (solution_of_tables T)
                     (fun n => mem_node n cannot_change) code_deps
                     preserve_direct_calls (static_names code_ids_bound e)
                     (bound_code_metadata e) unboxed
                     (fresh_name (names_of e)) next noted deps cds
                     (unit_kept e) (unit_codes e)
                     (static_names symbols_of_binding e) is_local Hcl Hslots
                     Hfields_read)
              e empty_unbox_env [] (unit_conts u) uinv_empty (Forall2_nil _)
              Hloc' Hedges)
    as (Hub & _ & _).
  exact (ub_refines unboxed Hdistinct _ _ Hub).
Qed.

(* Refinement up to observation, then refinement, is refinement up to
   observation. *)
Lemma refines_observably_refines (s m t : state) :
  refines_observably s m -> refines m t -> refines_observably s t.
Proof.
  intros (Hexit & Hback & Hwrong) (Hexit' & Hback').
  split; [ | split ].
  - intros k vs hp Hs. destruct (Hexit k vs hp Hs) as (vs' & Hm & Hvs).
    exists vs'. split; [ exact (Hexit' k vs' hp Hm) | exact Hvs ].
  - intros k vs' hp Ht.
    destruct (Hback' (Exits k vs' hp) ltac:(discriminate) Ht) as [Hm | Hm].
    + exact (Hback k vs' hp Hm).
    + right. exact (Hwrong Hm).
  - intros Ht. destruct (Hback' Goes_wrong ltac:(discriminate) Ht) as [Hm | Hm];
      exact (Hwrong Hm).
Qed.

Theorem rebuild_unbox_refines_observably :
  rebuild_unbox_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls Hlocal_slots Hprojections Hfragment Hstatic Hkinded
    Hwell_formed Hone Hclosed Hexact Hlocal.
  apply (refines_observably_refines _
           (State (rebuild_unit_from_tables (traverse_unit flag u e) T
                     preserve_direct_calls e)
              [] [] empty_static_env empty_heap)).
  - exact (rebuild_refines_observably flag is_local u e T
             preserve_direct_calls Hlocal_slots Hprojections Hfragment Hstatic
             Hkinded Hwell_formed Hone Hclosed Hexact).
  - exact (rebuild_unbox_refines_rebuild flag is_local u e T cannot_change
             (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls
             Hlocal_slots Hfragment Hclosed Hlocal).
Qed.

Theorem local_example : local_example_statement.
Proof.
  split.
  { intros f Hf. discriminate Hf. }
  do 12 (split; [ vm_compute; reflexivity | ]).
  split; exists 20; vm_compute; reflexivity.
Qed.

Theorem fragment_examples : fragment_examples_statement.
Proof. vm_compute. auto. Qed.

Theorem continuation_example : continuation_example_statement.
Proof.
  do 12 (split; [ vm_compute; reflexivity | ]).
  split; exists 20; vm_compute; reflexivity.
Qed.

Theorem spurious_keep_example : spurious_keep_example_statement.
Proof.
  do 10 (split; [ vm_compute; reflexivity | ]).
  split; exists 20; vm_compute; reflexivity.
Qed.

Theorem alias_example : alias_example_statement.
Proof.
  do 12 (split; [ vm_compute; reflexivity | ]).
  split; exists 20; vm_compute; reflexivity.
Qed.

Theorem loop_example : loop_example_statement.
Proof.
  do 12 (split; [ vm_compute; reflexivity | ]).
  split; exists 40; vm_compute; reflexivity.
Qed.

Theorem nested_example : nested_example_statement.
Proof.
  do 12 (split; [ vm_compute; reflexivity | ]).
  split; exists 20; vm_compute; reflexivity.
Qed.
