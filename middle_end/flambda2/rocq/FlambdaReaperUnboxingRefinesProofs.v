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
   FlambdaReaperRebuildProofs.v compares with the unit. The two rebuilt
   terms are related by [ub]: equal but at the [Let] of an unboxed
   variable, which the unboxing makes [Let]s of the variables of its
   fields ([lets]), and at a read of a field of it, which it makes the
   variable of the field. Related terms run alike ([ub_refines]), by a
   simulation of FlambdaReaperRebuildProofs.v ([shrinking_refines]): a
   step of the plain term is answered by as many steps as the unboxing
   makes [Let]s, none where it makes none; an unboxed variable bound in
   the plain run to a value has the variables of its fields bound in the
   other to the fields of the value ([env_rel]). Immutable blocks and
   boxed numbers are values, so no heap is involved.

   The two rebuilds of a unit of the fragment are related ([rebuild_ub]).
   What the plain rebuild keeps of an allocation that is unboxed, it keeps
   because the solution is closed ([field_kept]): each field unboxed is
   read with a use from a use of the variable ([get_fields_in] and the
   lemmas of "Where the unboxed fields come from"), the graph of the unit
   reads no function slot and makes no call ([closure_free_reads_blocks])
   so that each such use is a use of the variable, and the facts of the
   allocation are those of FlambdaReaperFlowProofs.expr_edges. The
   variables of the fields are fresh ([unit_unboxed_fields_fresh]), so
   that the checks of the fragment hold with "plain" read as "neither
   unboxed nor the variable of a field" ([strong_local]), which is what
   the relation asks. The statements compose. *)

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
    (kind : flambda_kind) (d : named) (fields : unboxed_fields) : bool :=
  match kind, d with
  | Value, Prim (Variadic (Make_block (Block_kind_values _))) args =>
    forallb plain_simple args
    && forallb
         (fun '(f, u) =>
            match f, u with
            | Field_block i Value, Not_unboxed _ =>
              match nth_error args i with
              | Some arg => kind_eqb (simple_kind (kinds_env kinds) arg) Value
              | None => false
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

Definition plain_apply_cont (ac : apply_cont) : bool :=
  forallb plain_simple (ac_args ac).

Definition local_binding (binding : static_binding) : bool :=
  match binding with
  | Bind_symbol _ (Static_const_block _ fields) =>
    forallb plain_simple fields
  | _ => false
  end.

Fixpoint local_expr (kinds : finite_map flambda_kind) (e : expr) : bool :=
  match e with
  | Let x kind d body =>
    negb (among x field_variables)
    && match fields_of unboxed x with
       | Some fields => unboxed_binding_ok kinds kind d fields
       | None => plain_binding_ok x kind d
       end
    && local_expr ((x, kind) :: kinds) body
  | Let_static bindings body =>
    forallb local_binding bindings && local_expr kinds body
  | Let_cont_non_recursive _ h body =>
    local_handler kinds h && local_expr kinds body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => local_handler kinds h) handlers
    && local_expr kinds body
  | Apply _ => false
  | Apply_cont ac => plain_apply_cont ac
  | Switch scrutinee arms =>
    plain_simple scrutinee && forallb (fun '(_, ac) => plain_apply_cont ac) arms
  | Invalid => true
  end
with local_handler (kinds : finite_map flambda_kind)
    (h : continuation_handler) : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb plain_variable params
    && local_expr (rev (combine params (hm_params_kinds metadata)) ++ kinds)
         handler
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

Inductive ub : expr -> expr -> Prop :=
  | ub_let (x : variable) (kind : flambda_kind) (d : named)
      (body body' : expr) :
    plain_variable M x = true -> named_plain d = true -> ub body body' ->
    ub (Let x kind d body) (Let x kind d body')
  | ub_alloc (x : variable) (d : named) (fields : unboxed_fields)
      (l : list (variable * flambda_kind * simple)) (body body' : expr) :
    fields_of M x = Some fields ->
    among x (field_variables M) = false ->
    Forall2
      (fun (fu : field * unboxed_field)
           (vks : variable * flambda_kind * simple) =>
         let '(f, u) := fu in
         let '(v, kind, s) := vks in
         u = Not_unboxed v ∧ alloc_field d f kind s
         ∧ plain_simple M s = true)
      fields l ->
    ub body body' ->
    ub (Let x Value d body) (lets l body')
  | ub_read (x : variable) (kind : flambda_kind) (d : named) (f : field)
      (s : simple) (y v : variable) (fields : unboxed_fields)
      (body body' : expr) :
    plain_variable M x = true ->
    read_of d = Some (f, s) ->
    (s = Var y ∨ ∃ k : flambda_kind, s = Const (Poison k)) ->
    fields_of M y = Some fields ->
    (f, Not_unboxed v) ∈ fields ->
    ub body body' ->
    ub (Let x kind d body) (Let x kind (Simple (Var v)) body')
  | ub_static (bindings : list static_binding) (body body' : expr) :
    forallb (local_binding M) bindings = true -> ub body body' ->
    ub (Let_static bindings body) (Let_static bindings body')
  | ub_let_cont (k : continuation) (params : list variable)
      (metadata : handler_metadata) (handler handler' body body' : expr) :
    forallb (plain_variable M) params = true -> ub handler handler' ->
    ub body body' ->
    ub (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k (Handler params metadata handler') body')
  | ub_let_cont_rec
      (handlers handlers' : list (continuation * continuation_handler))
      (body body' : expr) :
    ub_handlers handlers handlers' ->
    handlers_bind_statics handlers = handlers_bind_statics handlers' ->
    ub body body' ->
    ub (Let_cont_recursive handlers body) (Let_cont_recursive handlers' body')
  | ub_apply_cont (ac : apply_cont) :
    plain_apply_cont M ac = true -> ub (Apply_cont ac) (Apply_cont ac)
  | ub_switch (scrutinee : simple) (arms : list (Z * apply_cont)) :
    plain_simple M scrutinee = true ->
    forallb (fun '(_, ac) => plain_apply_cont M ac) arms = true ->
    ub (Switch scrutinee arms) (Switch scrutinee arms)
  | ub_invalid : ub Invalid Invalid
with ub_handlers : list (continuation * continuation_handler)
    -> list (continuation * continuation_handler) -> Prop :=
  | ub_handlers_nil : ub_handlers [] []
  | ub_handlers_cons (k : continuation) (params : list variable)
      (metadata : handler_metadata) (handler handler' : expr)
      (handlers handlers' : list (continuation * continuation_handler)) :
    forallb (plain_variable M) params = true -> ub handler handler' ->
    ub_handlers handlers handlers' ->
    ub_handlers ((k, Handler params metadata handler) :: handlers)
      ((k, Handler params metadata handler') :: handlers').

(* The environments: a variable of neither kind has one value in both;
   an unboxed variable bound in the first has each of its fields unboxed
   bound in the second to the field of its value. *)
Definition env_rel (vars vars' : env) : Prop :=
  (∀ z : variable, plain_variable M z = true -> lookup z vars = lookup z vars')
  ∧ (∀ (x : variable) (fields : unboxed_fields) (w : value),
       fields_of M x = Some fields -> lookup x vars = Some w ->
       ∀ (f : field) (v : variable),
         (f, Not_unboxed v) ∈ fields ->
         ∃ w' : value, field_value f w = Some w' ∧ lookup v vars' = Some w').

Inductive cont_rel : cont_value -> cont_value -> Prop :=
  | cont_rel_outer (k : continuation) : cont_rel (Outer k) (Outer k)
  | cont_rel_closure (params : list variable) (metadata : handler_metadata)
      (handler handler' : expr) (vars vars' : env) (conts conts' : cont_env)
      (recursive recursive' : list (continuation * continuation_handler)) :
    forallb (plain_variable M) params = true -> ub handler handler' ->
    env_rel vars vars' -> conts_rel conts conts' ->
    ub_handlers recursive recursive' ->
    cont_rel (Closure (Handler params metadata handler) vars conts recursive)
      (Closure (Handler params metadata handler') vars' conts' recursive')
with conts_rel : cont_env -> cont_env -> Prop :=
  | conts_rel_nil : conts_rel [] []
  | conts_rel_cons (k : continuation) (c c' : cont_value)
      (conts conts' : cont_env) :
    cont_rel c c' -> conts_rel conts conts' ->
    conts_rel ((k, c) :: conts) ((k, c') :: conts').

Definition related (s t : state) : Prop :=
  ub (state_expr s) (state_expr t) ∧ env_rel (state_vars s) (state_vars t)
  ∧ conts_rel (state_conts s) (state_conts t)
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
  intros [Hplain Hfields] Hx. split.
  - intros z Hz. rewrite !lookup_cons. destruct (Pos.eqb z x);
      [ reflexivity | exact (Hplain z Hz) ].
  - intros y fields w0 Hy Hlookup f v Hin.
    rewrite lookup_other in Hlookup
      by (intros ->; exact (plain_not_unboxed M x fields Hx Hy)).
    destruct (Hfields y fields w0 Hy Hlookup f v Hin) as (w' & Hw' & Hv).
    exists w'. split; [ exact Hw' | ].
    rewrite lookup_other; [ exact Hv | ].
    intros ->. exact (plain_not_field M x y fields f Hx Hy Hin).
Qed.

Lemma env_rel_bind_params (params : list variable) :
  ∀ (vs : list value) (vars vars' bound : env),
    env_rel M vars vars' -> forallb (plain_variable M) params = true ->
    bind_params params vs vars = Some bound ->
    ∃ bound' : env,
      bind_params params vs vars' = Some bound' ∧ env_rel M bound bound'.
Proof.
  induction params as [ | p params IH]; intros [ | w vs] vars vars' bound
    Hrel Hparams Hbind; simpl in Hbind; try discriminate Hbind.
  - injection Hbind as <-. exists vars'. split; [ reflexivity | exact Hrel ].
  - simpl in Hparams. apply andb_true_iff in Hparams as [Hp Hparams].
    exact (IH vs _ _ bound (env_rel_bind vars vars' p w Hrel Hp) Hparams
             Hbind).
Qed.

Lemma env_rel_empty : env_rel M [] [].
Proof.
  split; [ reflexivity | ]. intros x fields w _ Hlookup. discriminate.
Qed.

(* A variable that is not plain and not unboxed is the variable of a
   field. *)

Lemma resolve_rel (conts conts' : cont_env) (k : continuation) :
  conts_rel M conts conts' -> cont_rel M (resolve k conts) (resolve k conts').
Proof.
  intros Hrel. unfold resolve.
  induction Hrel as [ | k' c c' conts conts' Hc _ IH]; simpl;
    [ constructor | ].
  destruct (Pos.eqb k k'); [ exact Hc | exact IH ].
Qed.

Lemma rec_closures_rel_from
    (all all' hs hs' : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) :
  ub_handlers M all all' -> ub_handlers M hs hs' -> env_rel M vars vars' ->
  conts_rel M conts conts' ->
  conts_rel M
    (map (fun '(k, h) => (k, Closure h vars conts all)) hs ++ conts)
    (map (fun '(k, h) => (k, Closure h vars' conts' all')) hs' ++ conts').
Proof.
  intros Hall Hhs Hvars Hconts.
  induction Hhs as [ | k params metadata h h' hs hs' Hp Hh _ IH];
    simpl; [ exact Hconts | ].
  constructor; [ | exact IH ].
  constructor; assumption.
Qed.

Lemma rec_closures_rel
    (handlers handlers' : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) :
  ub_handlers M handlers handlers' -> env_rel M vars vars' ->
  conts_rel M conts conts' ->
  conts_rel M (rec_closures handlers vars conts ++ conts)
    (rec_closures handlers' vars' conts' ++ conts').
Proof.
  intros Hh.
  exact (rec_closures_rel_from _ _ _ _ vars vars' conts conts' Hh Hh).
Qed.

End Envs.

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

Lemma related_answered (s t : state) :
  related M s t -> answered (related M) eq measure s t.
Proof.
  destruct s as [p vars conts statics hp], t as [q vars' conts' statics' hp'].
  intros (Hub & Hvars & Hconts & Hstatics & Hhp).
  cbn [state_expr state_vars state_conts state_statics state_heap] in *.
  subst statics' hp'. unfold answered.
  destruct Hub as [x kind d body body' Hx Hd Hbody
    | x d fields l body body' Hfields Hxf Hl Hbody
    | x kind d f s y v fields body body' Hx Hread Hs Hfields Hin Hbody
    | bindings body body' Hbindings Hbody
    | k params metadata h h' body body' Hparams Hh Hbody
    | handlers handlers' body body' Hhandlers Hstat Hbody
    | ac Hac
    | scrutinee arms Hscrutinee Harms
    | ].
  - (* A [Let] of both. *)
    rewrite step_let, (named_plain_eval M statics vars vars' d Hvars Hd).
    destruct (eval_named statics vars' d) as [w | ] eqn:Hw; [ | exact I ].
    destruct (kind_eqb (kind_of_value w) kind) eqn:Hk; [ | exact I ].
    right. left. eexists. split; [ rewrite step_let, Hw, Hk; reflexivity | ].
    split; [ exact Hbody | ].
    split; [ exact (env_rel_bind M _ _ x w Hvars Hx) | ].
    split; [ exact Hconts | split; reflexivity ].
  - (* The allocation of an unboxed variable. *)
    rewrite step_let.
    destruct (eval_named statics vars d) as [w | ] eqn:Hw; [ | exact I ].
    destruct (kind_eqb (kind_of_value w) Value) eqn:Hk; [ | exact I ].
    pose proof (alloc_entries_variables d fields l Hl) as Hvl.
    destruct (lets_run l body' vars' conts' statics hp)
      as (vars1 & Hrun & Hother & Hbound).
    + rewrite Hvl. exact (leaves_nodup M M_distinct x fields Hfields).
    + intros v kind s v' Hin Hv' ->.
      destruct (alloc_entries_l d fields l v kind (Var v') Hl Hin)
        as (f & _ & _ & Hplain).
      rewrite Hvl in Hv'. simpl in Hplain.
      apply plain_variable_iff in Hplain as [_ Hnot].
      pose proof (field_variables_in M x fields v' Hfields Hv') as Hfv.
      apply among_In in Hfv. rewrite Hfv in Hnot. discriminate.
    + intros v kind s Hin.
      destruct (alloc_entries_l d fields l v kind s Hl Hin)
        as (f & _ & Hf & Hplain).
      destruct (alloc_field_eval statics vars d w f kind s Hw Hf)
        as (w' & Hs & _ & Hkw).
      exists w'. split; [ | exact Hkw ].
      rewrite <- (plain_simple_eval M vars vars' s Hvars Hplain). exact Hs.
    + apply (run_answer (related M) _ _ _ _ (length l) Hrun).
      * unfold related.
        cbn [state_expr state_vars state_conts state_statics state_heap].
        split; [ exact Hbody | ]. split; [ | split; [ exact Hconts | ] ];
          [ | split; reflexivity ].
        split.
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
              destruct (alloc_entries_r d fields l f v Hl Hfv)
                as (kind & s & Hin & Hf & Hplain).
              destruct (alloc_field_eval statics vars d w f kind s Hw Hf)
                as (w' & Hs & Hfw & _).
              exists w'. split; [ exact Hfw | ].
              apply (Hbound v kind s w' Hin).
              rewrite <- (plain_simple_eval M vars vars' s Hvars Hplain).
              exact Hs.
           ++ rewrite lookup_other in Hlookup by exact Hne.
              destruct (proj2 Hvars y fy w0 Hy Hlookup f v Hfv)
                as (w' & Hfw & Hv).
              exists w'. split; [ exact Hfw | ]. rewrite Hother; [ exact Hv | ].
              rewrite Hvl. intros Hvx.
              exact (leaves_apart M M_distinct y x fy fields v Hy Hfields Hne
                       (leaf_in fy f v Hfv) Hvx).
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
    destruct (proj2 Hvars y fields w0 Hfields Hw0 f v Hin)
      as (w' & Hfw & Hv).
    rewrite <- (Hfield w' Hfw) in Hv.
    right. left. eexists. split.
    + rewrite step_let. cbn [eval_named eval_simple]. rewrite Hv, Hk.
      reflexivity.
    + split; [ exact Hbody | ].
      split; [ exact (env_rel_bind M _ _ x r Hvars Hx) | ].
      split; [ exact Hconts | split; reflexivity ].
  - (* A [Let_static]. *)
    rewrite step_let_static,
      (local_bindings_eval M vars vars' bindings Hvars Hbindings statics).
    destruct (eval_static_bindings vars' bindings statics) as [statics1 | ]
      eqn:Hst; [ | exact I ].
    right. left. eexists.
    split; [ rewrite step_let_static, Hst; reflexivity | ].
    split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ exact Hconts | split; reflexivity ].
  - (* A non-recursive [Let_cont]. *)
    rewrite step_let_cont_non_recursive. right. left. eexists.
    split; [ rewrite step_let_cont_non_recursive; reflexivity | ].
    split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ | split; reflexivity ].
    constructor; [ | exact Hconts ].
    constructor; [ exact Hparams | exact Hh | exact Hvars | exact Hconts | ].
    constructor.
  - (* A recursive [Let_cont]. *)
    rewrite step_let_cont_recursive.
    destruct (handlers_bind_statics handlers) eqn:Hb; [ exact I | ].
    right. left. eexists.
    split.
    { rewrite step_let_cont_recursive.
      destruct (handlers_bind_statics handlers') eqn:Hb';
        [ congruence | reflexivity ]. }
    split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ apply rec_closures_rel; assumption | split; reflexivity ].
  - (* An [Apply_cont]. *)
    rewrite step_apply_cont. unfold plain_apply_cont in Hac.
    rewrite (plain_simples_eval M vars vars' _ Hvars Hac).
    destruct (eval_simples vars' (ac_args ac)) as [vs | ] eqn:Hvs;
      [ | exact I ].
    unfold apply_continuation.
    pose proof (resolve_rel M conts conts' (ac_continuation ac) Hconts) as Hc.
    remember (resolve (ac_continuation ac) conts) as c eqn:Ec.
    remember (resolve (ac_continuation ac) conts') as c' eqn:Ec'.
    destruct Hc as [k | params metadata h h' cvars cvars' cconts cconts' rec
                      rec' Hp Hh Hcv Hcc Hrec].
    + cbn [apply_cont_value]. exists vs. split; [ | reflexivity ].
      rewrite step_apply_cont, Hvs. unfold apply_continuation.
      rewrite <- Ec'. reflexivity.
    + cbn [apply_cont_value].
      destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
        eqn:Hkinds; [ | exact I ].
      destruct (bind_params params vs cvars) as [hv | ] eqn:Hbind;
        [ | exact I ].
      destruct (env_rel_bind_params M params vs cvars cvars' hv Hcv Hp Hbind)
        as (hv' & Hbind' & Hhv).
      right. left. eexists. split.
      * rewrite step_apply_cont, Hvs. unfold apply_continuation.
        rewrite <- Ec'. cbn [apply_cont_value]. rewrite Hkinds, Hbind'.
        reflexivity.
      * split; [ exact Hh | split; [ exact Hhv | ] ].
        split; [ apply rec_closures_rel; assumption | split; reflexivity ].
  - (* A [Switch]. *)
    rewrite step_switch,
      (plain_simple_eval M vars vars' scrutinee Hvars Hscrutinee).
    destruct (eval_simple vars' scrutinee) as [w | ] eqn:Hw; [ | exact I ].
    destruct (naked_immediate_of_value w) as [i | ] eqn:Hi; [ | exact I ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | exact I ].
    right. left. eexists.
    split; [ rewrite step_switch, Hw, Hi, Harm; reflexivity | ].
    split; [ | split; [ exact Hvars | split; [ exact Hconts | ] ] ];
      [ | split; reflexivity ].
    constructor. apply FlambdaSemanticsProofs.find_arm_in in Harm.
    rewrite forallb_forall in Harms. exact (Harms _ Harm).
  - rewrite step_invalid. exact I.
Qed.

(* Related terms, run on their own, refine each other. *)
Lemma ub_refines (p q : expr) :
  ub M p q ->
  refines (State p [] [] empty_static_env empty_heap)
    (State q [] [] empty_static_env empty_heap).
Proof.
  intros Hub. apply (shrinking_refines (related M) eq measure).
  - exact related_answered.
  - intros vs vs' ->. reflexivity.
  - split; [ exact Hub | split; [ apply env_rel_empty | ] ].
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
    (kind : flambda_kind) (d : named) (fields : unboxed_fields) :
  FlambdaReaperUnboxingRefinesSpec.unboxed_binding_ok unboxed kinds kind d
    fields = true ->
  (∀ z, z ∈ variables_in_named d -> fresh z) ->
  unboxed_binding_ok unboxed kinds kind d fields = true.
Proof.
  unfold unboxed_binding_ok,
    FlambdaReaperUnboxingRefinesSpec.unboxed_binding_ok.
  intros Hd Hz. destruct kind; try discriminate Hd.
  destruct d as [ | p args | | | | | ]; try discriminate Hd.
  destruct p as [p | p | p | p]; try discriminate Hd.
  - destruct p; try discriminate Hd.
    destruct args as [ | c [ | ]]; try discriminate Hd.
    apply andb_true_iff in Hd as [Hc Hfields].
    rewrite (strong_simple c Hc); [ exact Hfields | ].
    intros z Hin. apply Hz. simpl. unfold variables_in_simples. simpl.
    rewrite app_nil_r. exact Hin.
  - destruct p as [[tag] | ]; try discriminate Hd.
    apply andb_true_iff in Hd as [Hargs Hfields].
    rewrite (strong_simples args Hargs Hz). exact Hfields.
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

Lemma strong_local :
  (∀ (e : expr) (kinds : finite_map flambda_kind),
     FlambdaReaperUnboxingRefinesSpec.local_expr unboxed sol kinds e = true ->
     (∀ z, z ∈ variables_of e -> fresh z) ->
     local_expr unboxed sol kinds e = true)
  ∧ (∀ (h : continuation_handler) (kinds : finite_map flambda_kind),
       FlambdaReaperUnboxingRefinesSpec.local_handler unboxed sol kinds h
       = true ->
       (∀ z, z ∈ variables_in_handler h -> fresh z) ->
       local_handler unboxed sol kinds h = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH kinds Hloc Hz. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hd Hbody].
    rewrite (Hz x (or_introl eq_refl)). simpl.
    apply andb_true_iff. split.
    + destruct (fields_of unboxed x) as [fields | ].
      * apply (strong_unboxed_binding _ _ _ _ Hd).
        intros z Hin. apply Hz. right. apply in_or_app. left. exact Hin.
      * apply (strong_plain_binding _ _ _ Hd).
        intros z Hin. apply Hz. right. apply in_or_app. left. exact Hin.
    + apply (IH _ Hbody). intros z Hin. apply Hz. right. apply in_or_app.
      right. exact Hin.
  - intros bindings body _ IH kinds Hloc Hz. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hbs Hbody].
    apply andb_true_iff. split.
    + apply forallb_forall. intros b Hb. rewrite forallb_forall in Hbs.
      specialize (Hbs b Hb).
      destruct b as [id c | id | sym [ | | tag fields | | | ]];
        try discriminate Hbs.
      apply (strong_simples fields Hbs). intros z Hin. apply Hz.
      apply in_or_app. left. apply in_flat_map. exists (Bind_symbol sym
        (Static_const_block tag fields)). split; [ exact Hb | exact Hin ].
    + apply (IH _ Hbody). intros z Hin. apply Hz. apply in_or_app. right.
      exact Hin.
  - intros k h body IHh IH kinds Hloc Hz. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hh Hbody].
    apply andb_true_iff. split.
    + apply (IHh _ Hh). intros z Hin. apply Hz. apply in_or_app. left.
      exact Hin.
    + apply (IH _ Hbody). intros z Hin. apply Hz. apply in_or_app. right.
      exact Hin.
  - intros handlers body IHh IH kinds Hloc Hz. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hhs Hbody].
    apply andb_true_iff. split.
    + apply forallb_forall. intros [k h] Hin. rewrite forallb_forall in Hhs.
      apply (IHh k h Hin _ (Hhs _ Hin)). intros z Hz'. apply Hz.
      apply in_or_app. left. apply in_flat_map. exists (k, h).
      split; [ exact Hin | exact Hz' ].
    + apply (IH _ Hbody). intros z Hin. apply Hz. apply in_or_app. right.
      exact Hin.
  - intros ap kinds Hloc. discriminate Hloc.
  - intros ac kinds Hloc Hz. simpl in Hloc |- *.
    exact (strong_simples _ Hloc Hz).
  - intros scrutinee arms kinds Hloc Hz. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hs Harms].
    apply andb_true_iff. split.
    + apply (strong_simple _ Hs). intros z Hin. apply Hz. apply in_or_app.
      left. exact Hin.
    + apply forallb_forall. intros [d ac] Hin. rewrite forallb_forall in Harms.
      apply (strong_simples _ (Harms _ Hin)). intros z Hz'. apply Hz.
      apply in_or_app. right. apply in_flat_map. exists (d, ac).
      split; [ exact Hin | exact Hz' ].
  - intros kinds _ _. reflexivity.
  - intros params metadata handler IH kinds Hloc Hz. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hloc Hbody].
    apply andb_true_iff in Hloc as [Hlen Hparams].
    rewrite Hlen. simpl. apply andb_true_iff. split.
    + apply forallb_forall. intros p Hp. rewrite forallb_forall in Hparams.
      apply (strong_variable p (Hparams p Hp)). apply Hz. apply in_or_app.
      left. exact Hp.
    + apply (IH _ Hbody). intros z Hin. apply Hz. apply in_or_app. right.
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
  (∀ (e : expr) (kinds : finite_map flambda_kind),
     FlambdaReaperUnboxingRefinesSpec.local_expr unboxed sol kinds e = true
     -> closure_free e = true)
  ∧ (∀ (h : continuation_handler) (kinds : finite_map flambda_kind),
       FlambdaReaperUnboxingRefinesSpec.local_handler unboxed sol kinds h
       = true ->
       handler_closure_free h = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH kinds Hloc. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hd Hbody].
    rewrite (IH _ Hbody), andb_true_r.
    destruct (fields_of unboxed x) as [fields | ].
    + unfold FlambdaReaperUnboxingRefinesSpec.unboxed_binding_ok in Hd.
      destruct kind; try discriminate Hd. destruct d; try discriminate Hd.
      reflexivity.
    + unfold FlambdaReaperUnboxingRefinesSpec.plain_binding_ok in Hd.
      destruct (unboxed_read unboxed d) as [[f fields] | ] eqn:Hread.
      * unfold unboxed_read in Hread. destruct d; try discriminate Hread.
        reflexivity.
      * destruct d; try discriminate Hd; reflexivity.
  - intros bindings body _ IH kinds Hloc. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hbs Hbody].
    rewrite (IH _ Hbody), andb_true_r. apply forallb_forall. intros b Hb.
    rewrite forallb_forall in Hbs. specialize (Hbs b Hb).
    destruct b as [ | | s [ | | | | | ]]; try discriminate Hbs. reflexivity.
  - intros k h body IHh IH kinds Hloc. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hh Hbody].
    rewrite (IHh _ Hh), (IH _ Hbody). reflexivity.
  - intros handlers body IHh IH kinds Hloc. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hhs Hbody].
    rewrite (IH _ Hbody), andb_true_r. apply forallb_forall.
    intros [k h] Hin. rewrite forallb_forall in Hhs.
    exact (IHh k h Hin _ (Hhs _ Hin)).
  - intros ap kinds Hloc. discriminate Hloc.
  - intros ac kinds _. reflexivity.
  - intros scrutinee arms kinds _. reflexivity.
  - intros kinds _. reflexivity.
  - intros params metadata handler IH kinds Hloc. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [_ Hbody]. exact (IH _ Hbody).
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
    (f : field) (v : variable),
  fields_of unboxed x = Some fields -> (f, Not_unboxed v) ∈ fields ->
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
    (v : variable) (a : simple) (en : rebuild_env) :
  fields_of unboxed x = Some fields -> (f, Not_unboxed v) ∈ fields ->
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

(* What the rebuild with unboxing knows is what the plain rebuild knows:
   its continuations have the masks of their decisions, which unbox
   nothing; and no projection is pending. *)
Record uinv (en : unbox_env) : Prop := {
  uinv_drop : ue_drop_projections en = false;
  uinv_projections : env_projections (ue_env en) = [];
  uinv_conts : ∀ k : continuation,
    lookup k (env_conts (ue_env en))
    = option_map (map is_kept_decision) (lookup k (ue_decisions en));
  uinv_decided : ∀ (k : continuation) (ds : list decision),
    lookup k (ue_decisions en) = Some ds -> forallb is_decided ds = true
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

Lemma params_of_kept_decisions (bound : list (variable * flambda_kind)) :
  ∀ mask : list bool,
    length mask = length bound ->
    params_of_decisions bound
      (map
         (fun p : (variable * flambda_kind) * bool =>
            let '((x, kind), keep) := p in
            Decided (if keep then Keep kind else Delete))
         (combine bound mask))
    = filter_kept mask bound.
Proof.
  induction bound as [ | [x kind] bound IH]; intros [ | keep mask] Hlen;
    simpl in Hlen |- *; try discriminate Hlen; [ reflexivity | ].
  rewrite IH by lia. destruct keep; reflexivity.
Qed.

Lemma params_to_keep_length (is_exn : bool)
    (bound : list (variable * flambda_kind)) :
  length (params_to_keep sol is_exn bound) = length bound.
Proof. unfold params_to_keep. apply length_map. Qed.

(* The decisions of a handler whose parameters are not unboxed: those of
   its mask. *)
Lemma handler_decisions_plain (is_exn : bool) (params : list variable)
    (metadata : handler_metadata) (handler : expr) :
  forallb (plain_variable unboxed) params = true ->
  handler_decisions sol unboxed is_exn (Handler params metadata handler)
  = map
      (fun p : (variable * flambda_kind) * bool =>
         let '((x, kind), keep) := p in
         Decided (if keep then Keep kind else Delete))
      (combine (with_kinds params (hm_params_kinds metadata))
         (params_to_keep sol is_exn
            (with_kinds params (hm_params_kinds metadata)))).
Proof.
  intros Hparams. unfold handler_decisions. apply map_ext_in.
  intros [[x kind] keep] Hin. apply in_combine_l in Hin.
  assert (Hx : x ∈ params).
  { clear Hparams. revert Hin. generalize (hm_params_kinds metadata).
    induction params as [ | p params IH]; intros kinds Hin;
      [ destruct Hin | ].
    destruct kinds as [ | k kinds]; simpl in Hin;
      (destruct Hin as [[= -> _] | Hin]; [ left; reflexivity | ]);
      right; exact (IH _ Hin). }
  rewrite forallb_forall in Hparams. specialize (Hparams x Hx).
  apply plain_variable_iff in Hparams as [Hx' _]. unfold Keyed.is_bound in Hx'.
  unfold unboxed_of_node. destruct (Keyed.lookup _ _ _); [ discriminate | ].
  reflexivity.
Qed.

Lemma kept_decisions_mask (bound : list (variable * flambda_kind)) :
  ∀ mask : list bool,
    length mask = length bound ->
    map is_kept_decision
      (map
         (fun p : (variable * flambda_kind) * bool =>
            let '((x, kind), keep) := p in
            Decided (if keep then Keep kind else Delete))
         (combine bound mask))
    = mask.
Proof.
  induction bound as [ | [x kind] bound IH]; intros [ | keep mask] Hlen;
    simpl in Hlen |- *; try discriminate Hlen; [ reflexivity | ].
  rewrite IH by lia. destruct keep; reflexivity.
Qed.

Lemma kept_decisions_decided (bound : list (variable * flambda_kind))
    (mask : list bool) :
  forallb is_decided
    (map
       (fun p : (variable * flambda_kind) * bool =>
          let '((x, kind), keep) := p in
          Decided (if keep then Keep kind else Delete))
       (combine bound mask))
  = true.
Proof.
  apply forallb_forall. intros d Hd. apply in_map_iff in Hd.
  destruct Hd as ([[x kind] keep] & <- & _). reflexivity.
Qed.

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

Lemma rewrite_apply_cont_same (en : unbox_env) (ac : apply_cont) :
  uinv en -> plain_apply_cont unboxed ac = true ->
  rewrite_apply_cont_u sol unboxed en ac
  = rewrite_apply_cont sol (ue_env en) ac.
Proof.
  intros Hen Hac. unfold rewrite_apply_cont_u, rewrite_apply_cont.
  destruct (existsb (simple_has_no_source sol) (ac_args ac));
    [ reflexivity | ].
  rewrite (uinv_conts en Hen).
  destruct (lookup (ac_continuation ac) (ue_decisions en)) as [ds | ] eqn:Hds;
    simpl.
  - rewrite (get_args_decided _ _ Hac ds (uinv_decided en Hen _ ds Hds)).
    reflexivity.
  - assert (Hmap : map (rewrite_simple sol_u (ue_env en)) (ac_args ac)
                   = map (rewrite_simple sol (ue_env en)) (ac_args ac)).
    { apply map_ext_in. intros s Hs. apply rewrite_simple_u.
      unfold plain_apply_cont in Hac. rewrite forallb_forall in Hac.
      exact (Hac s Hs). }
    rewrite Hmap. reflexivity.
Qed.

Lemma rewrite_apply_cont_plain (en : rebuild_env) (ac ac' : apply_cont) :
  plain_apply_cont unboxed ac = true -> rewrite_apply_cont sol en ac = Some ac'
  -> plain_apply_cont unboxed ac' = true.
Proof.
  intros Hac. unfold rewrite_apply_cont.
  destruct (existsb (simple_has_no_source sol) (ac_args ac));
    [ discriminate | ].
  intros [= <-]. unfold plain_apply_cont in *. cbn [ac_args].
  pose proof (rewrite_simples_plain en _ Hac) as Hmap.
  destruct (lookup (ac_continuation ac) (env_conts en)) as [mask | ];
    [ | exact Hmap ].
  apply forallb_forall. intros s Hs. rewrite forallb_forall in Hmap.
  apply Hmap. clear -Hs. revert mask Hs.
  induction (map (rewrite_simple sol en) (ac_args ac)) as [ | a l IH];
    intros [ | keep mask] Hs; simpl in Hs; try contradiction.
  - destruct Hs as [<- | Hs]; [ left; reflexivity | right ].
    exact (IH [] Hs).
  - destruct keep; [ destruct Hs as [<- | Hs]; [ left; reflexivity | ] | ];
      right; exact (IH mask Hs) || exact (IH mask Hs).
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

Lemma uinv_bind (en : unbox_env) (bound : list (variable * flambda_kind)) :
  uinv en -> uinv (ubind_variables en bound).
Proof.
  intros [H1 H2 H3 H4]. split; [ reflexivity | reflexivity | exact H3 | ].
  exact H4.
Qed.

Lemma lookup_map_values {A B : Type} (f : A -> B) (k : positive)
    (l : list (positive * A)) :
  lookup k (map (fun '(k', a) => (k', f a)) l) = option_map f (lookup k l).
Proof.
  induction l as [ | [k' a] l IH]; [ reflexivity | ]. cbn [map].
  rewrite !lookup_cons. destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

Lemma uinv_continuations (en : unbox_env)
    (bound : list (continuation * list decision)) :
  uinv en ->
  (∀ (k : continuation) (ds : list decision),
     (k, ds) ∈ bound -> forallb is_decided ds = true) ->
  uinv (ubind_continuations en bound).
Proof.
  intros [H1 H2 H3 H4] Hbound. split; [ reflexivity | reflexivity | | ].
  - intros k. cbn [ubind_continuations ue_env ue_decisions bind_continuations
                   env_conts].
    rewrite !lookup_app, lookup_map_values.
    destruct (lookup k bound); [ reflexivity | exact (H3 k) ].
  - intros k ds. cbn [ubind_continuations ue_decisions].
    rewrite lookup_app. destruct (lookup k bound) as [ds' | ] eqn:Hk.
    + intros [= <-]. apply (Hbound k). apply lookup_in, Hk.
    + apply H4.
Qed.

(* The [Let] of an unboxed variable, as the fragment has it. *)
Lemma unboxed_binding_cases (kinds : finite_map flambda_kind)
    (kind : flambda_kind) (d : named) (fields : unboxed_fields) :
  unboxed_binding_ok unboxed kinds kind d fields = true ->
  kind = Value
  ∧ ((∃ (tag : Z) (args : list simple),
        d = Prim (Variadic (Make_block (Block_kind_values tag))) args
        ∧ forallb (plain_simple unboxed) args = true
        ∧ ∀ (f : field) (u : unboxed_field),
            (f, u) ∈ fields ->
            ∃ (i : nat) (v : variable) (a : simple),
              f = Field_block i Value ∧ u = Not_unboxed v
              ∧ nth_error args i = Some a
              ∧ simple_kind (kinds_env kinds) a = Value)
     ∨ (∃ (bn : boxable_number) (c : simple),
          d = Prim (Unary (Box_number bn)) [c]
          ∧ plain_simple unboxed c = true
          ∧ ∀ (f : field) (u : unboxed_field),
              (f, u) ∈ fields ->
              ∃ v : variable, f = Field_boxed_number bn ∧ u = Not_unboxed v)).
Proof.
  unfold unboxed_binding_ok. intros H.
  destruct kind; try discriminate H. split; [ reflexivity | ].
  destruct d as [ | p args | | | | | ]; try discriminate H.
  destruct p as [p | p | p | p]; try discriminate H.
  - destruct p; try discriminate H. rename kind into bn.
    destruct args as [ | c [ | ]]; try discriminate H.
    right. apply andb_true_iff in H as [Hc Hfields].
    exists bn, c. split; [ reflexivity | split; [ exact Hc | ] ].
    intros f u Hin. rewrite forallb_forall in Hfields.
    specialize (Hfields (f, u) Hin). cbn beta iota in Hfields.
    destruct u as [v | ]; [ | discriminate Hfields ].
    apply field_eqb_eq in Hfields. subst f.
    exists v. split; reflexivity.
  - destruct p as [[tag] | ]; try discriminate H.
    left. apply andb_true_iff in H as [Hargs Hfields].
    exists tag, args. split; [ reflexivity | split; [ exact Hargs | ] ].
    intros f u Hin. rewrite forallb_forall in Hfields.
    specialize (Hfields (f, u) Hin). cbn beta iota in Hfields.
    destruct f as [i k | | | | | | | | ]; try discriminate Hfields.
    destruct k; try discriminate Hfields.
    destruct u as [v | ]; [ | discriminate Hfields ].
    destruct (nth_error args i) as [a | ] eqn:Hnth; [ | discriminate Hfields ].
    apply kind_eqb_eq in Hfields.
    exists i, v, a. repeat split; assumption.
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

Definition ub_result (p q : expr) : Prop :=
  ub unboxed p q
  ∧ free_continuations_of_expr p = free_continuations_of_expr q
  ∧ binds_statics p = binds_statics q.

Lemma ub_result_invalid : ub_result Invalid Invalid.
Proof. split; [ constructor | split; reflexivity ]. Qed.

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

Lemma local_bindings_rebuilt (en : unbox_env)
    (bindings : list static_binding) :
  forallb (local_binding unboxed) bindings = true ->
  flat_map
    (rebuild_binding_u F sol cannot_change code_deps preserve_direct_calls
       unit_code_ids codes unboxed first_fresh_u [] en)
    bindings
  = flat_map
      (rebuild_binding F sol preserve_direct_calls unit_code_ids first_fresh
         (ue_env en))
      bindings
  ∧ forallb (local_binding unboxed)
      (flat_map
         (rebuild_binding F sol preserve_direct_calls unit_code_ids
            first_fresh (ue_env en))
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
  split; [ reflexivity | ]. rewrite Hplain, andb_true_r.
  apply rebuild_fields_plain, Hb.
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

Lemma rebuild_ub :
  (∀ e : expr,
     ∀ (en : unbox_env) (ce : finite_map (list node)),
       uinv en ->
       local_expr unboxed sol (env_kinds (ue_env en)) e = true ->
       edges ce e ->
       ub_result (plain_rebuild (ue_env en) e) (unbox_rebuild en e))
  ∧ (∀ h : continuation_handler,
       ∀ (en : unbox_env) (ce : finite_map (list node)),
         uinv en ->
         local_handler unboxed sol (env_kinds (ue_env en)) h = true ->
         FlambdaReaperFlowProofs.handler_edges (graph_of_facts F) noted deps
           cds kept_slots codes0 symbols True None ce h ->
         ∀ dead is_exn : bool,
           ∃ (params : list variable) (metadata : handler_metadata)
             (hp hu : expr),
             rebuild_handler F sol preserve_direct_calls unit_code_ids
               first_fresh (ue_env en) dead (handler_mask sol is_exn h) h
             = Handler params metadata hp
             ∧ rebuild_handler_u F sol cannot_change code_deps
                 preserve_direct_calls unit_code_ids codes unboxed
                 first_fresh_u [] en dead
                 (handler_decisions sol unboxed is_exn h) h
               = Handler params metadata hu
             ∧ forallb (plain_variable unboxed) params = true
             ∧ ub_result hp hu).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - (* Let *)
    intros x kind d body IH en ce Hen Hloc Hedges. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hloc Hbody].
    apply andb_true_iff in Hloc as [Hxf Hbind]. apply negb_true_iff in Hxf.
    destruct Hedges as [Hnamed_edges Hbody_edges].
    pose proof (uinv_bind en [(x, kind)] Hen) as Hen'.
    specialize (IH (ubind_variables en [(x, kind)]) ce Hen' Hbody
                  Hbody_edges).
    unfold plain_rebuild, unbox_rebuild in *. cbn [rebuild_expr rebuild_expr_u].
    unfold is_pending_projection. rewrite (uinv_projections en Hen).
    assert (Hd : (∃ s, d = Simple s) ∨ (∃ p args, d = Prim p args)).
    { destruct (fields_of unboxed x) as [fields | ] eqn:Hx.
      - destruct (unboxed_binding_cases _ kind d fields Hbind)
          as (_ & [(tag & args & -> & _) | (bn & c & -> & _)]);
          right; eauto.
      - destruct (plain_binding_cases x kind d Hbind)
          as [Hp | (p & y & f & fs & -> & _)]; [ | right; eauto ].
        destruct d; try discriminate Hp; eauto. }
    destruct IH as (IHub & IHfc & IHbs).
    cbn [ue_env ubind_variables ubind_variables_then ubind_set] in IHub, IHfc,
      IHbs.
    destruct Hd as [(s0 & ->) | (p & args & ->)]; cbn beta iota;
      (destruct (is_dead_var sol x kind); [ exact ub_result_invalid | ]);
      (destruct (is_var_used sol x kind) eqn:Hused;
       [ | split; [ exact IHub | split; assumption ] ]);
      change (unboxed_of_node unboxed (Node_variable x))
        with (fields_of unboxed x);
      change (changed_of_node [] (Node_variable x)) with (@None closure_repr);
      cbn iota;
      (destruct (fields_of unboxed x) as [fields | ] eqn:Hx).
    + (* An unboxed simple: not in the fragment. *)
      destruct (unboxed_binding_cases _ kind _ fields Hbind)
        as (_ & [(tag & args & Heq & _) | (bn & c & Heq & _)]);
        discriminate Heq.
    + (* A simple. *)
      destruct (plain_binding_cases x kind _ Hbind)
        as [Hp | (p & y & f & fs & Heq & _)]; [ | discriminate Heq ].
      rewrite (rebuild_let_u_plain _ x kind _ _ Hp).
      split; [ | split; assumption ].
      constructor; [ | apply named_plain_rebuilt, Hp | exact IHub ].
      apply plain_variable_iff. unfold Keyed.is_bound.
      unfold fields_of in Hx. rewrite Hx. split; [ reflexivity | exact Hxf ].
    + (* An unboxed allocation. *)
      destruct (unboxed_binding_cases _ kind _ fields Hbind)
        as (-> & [(tag & args' & Heq & Hargs & Hfields)
                 | (bn & c & Heq & Hc & Hfields)]);
        injection Heq as -> ->; simpl in Hnamed_edges.
      * destruct (bind_block_fields_lets (ue_env en) x tag args' fields Hargs)
          with (body := unbox_rebuild (ubind_variables en [(x, Value)]) body)
          as (l & Hl & Hentries).
        { intros f u Hin.
          destruct (Hfields f u Hin) as (i & v & a & -> & -> & Hnth & Hkind).
          rewrite kinds_env_kind in Hkind.
          destruct (field_kept x fields _ v a (ue_env en) Hx Hin
                      (proj1 Hnamed_edges i a Hnth) eq_refl)
            as [Hused' Hkept].
          exists i, v, a. repeat split; assumption. }
        unfold rebuild_unboxed_binding. unfold unbox_rebuild in Hl.
        rewrite Hl. cbn [rebuild_named].
        destruct (lets_free_continuations l
                    (unbox_rebuild (ubind_variables en [(x, Value)]) body))
          as [Hfc Hbs].
        unfold unbox_rebuild in Hfc, Hbs.
        split; [ exact (ub_alloc unboxed x _ fields l _ _ Hx Hxf Hentries IHub)
               | ].
        cbn [free_continuations_of_expr binds_statics]. rewrite Hfc, Hbs.
        split; assumption.
      * destruct (bind_boxed_number_lets bn c fields
                    (rebuild_named sol (ue_env en) x
                       (Prim (Unary (Box_number bn)) [c])) Hc)
          with (body := unbox_rebuild (ubind_variables en [(x, Value)]) body)
          as (l & Hl & Hentries).
        { intros Hne. destruct fields as [ | [f u] fields]; [ easy | ].
          destruct (Hfields f u (or_introl eq_refl)) as (v & -> & ->).
          destruct (field_kept x _ _ v c (ue_env en) Hx (or_introl eq_refl)
                      (proj1 Hnamed_edges) eq_refl)
            as [Hused' Hkept].
          cbn [rebuild_named].
          unfold field_used. rewrite Hused'. simpl.
          rewrite (kept_rewrite _ c Hkept). reflexivity. }
        { intros f u Hin. exact (Hfields f u Hin). }
        unfold rebuild_unboxed_binding. unfold unbox_rebuild in Hl.
        rewrite Hl.
        destruct (lets_free_continuations l
                    (unbox_rebuild (ubind_variables en [(x, Value)]) body))
          as [Hfc Hbs].
        unfold unbox_rebuild in Hfc, Hbs.
        split; [ exact (ub_alloc unboxed x _ fields l _ _ Hx Hxf Hentries IHub)
               | ].
        cbn [free_continuations_of_expr binds_statics]. rewrite Hfc, Hbs.
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
        apply (ub_read unboxed x kind
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
    intros bindings body _ IH en ce Hen Hloc Hedges. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hbs Hbody].
    pose proof (uinv_bind en [] Hen) as Hen'.
    destruct (IH (ubind_variables en []) ce Hen' Hbody (proj2 Hedges))
      as (IHub & IHfc & IHbs).
    destruct (local_bindings_rebuilt en bindings Hbs) as [Heq Hplain].
    rewrite plain_rebuild_let_static, unbox_rebuild_let_static, Heq.
    change (bind_variables (ue_env en) [])
      with (ue_env (ubind_variables en [])).
    split; [ constructor; [ exact Hplain | exact IHub ] | ].
    cbn [free_continuations_of_expr binds_statics]. rewrite IHfc, IHbs.
    split; reflexivity.
  - (* Let_cont_non_recursive *)
    intros k h body IHh IH en ce Hen Hloc Hedges. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hh Hbody].
    destruct Hedges as [Hh_edges Hbody_edges].
    destruct h as [params metadata handler].
    pose proof Hh as Hh'. simpl in Hh'.
    apply andb_true_iff in Hh' as [Hh' _].
    apply andb_true_iff in Hh' as [_ Hparams].
    rewrite plain_rebuild_let_cont, unbox_rebuild_let_cont. cbv zeta.
    set (is_exn := handler_is_exn_handler (Handler params metadata handler)).
    set (ds := handler_decisions sol unboxed is_exn
                 (Handler params metadata handler)).
    set (mask := handler_mask sol is_exn (Handler params metadata handler)).
    assert (Hmask : map is_kept_decision ds = mask).
    { unfold ds, mask. rewrite (handler_decisions_plain is_exn params metadata
                                 handler Hparams).
      apply kept_decisions_mask. unfold handler_mask.
      rewrite params_to_keep_length. reflexivity. }
    assert (Hdec : forallb is_decided ds = true).
    { unfold ds. rewrite handler_decisions_plain by exact Hparams.
      apply kept_decisions_decided. }
    assert (Hen' : uinv (ubind_continuations en [(k, ds)])).
    { apply uinv_continuations; [ exact Hen | ].
      intros k' ds' [[= <- <-] | []]. exact Hdec. }
    assert (IH' : ub_result
                    (plain_rebuild (bind_continuations (ue_env en) [(k, mask)])
                       body)
                    (unbox_rebuild (ubind_continuations en [(k, ds)]) body)).
    { rewrite <- Hmask. exact (IH _ _ Hen' Hbody Hbody_edges). }
    destruct IH' as (IHub & IHfc & IHbs).
    rewrite <- IHfc.
    destruct (among k _); [ | split; [ exact IHub | split; assumption ] ].
    destruct (IHh en ce Hen Hh Hh_edges true is_exn)
      as (ps & md & hp & hu & Hp & Hu & Hps & Hhub & Hhfc & Hhbs).
    fold mask in Hp. fold ds in Hu. rewrite Hp, Hu.
    split; [ constructor; assumption | ].
    cbn [free_continuations_of_expr free_continuations_of_handler binds_statics
         handler_binds_statics].
    rewrite Hhfc, IHfc, Hhbs, IHbs. split; reflexivity.
  - (* Let_cont_recursive *)
    intros handlers body IHh IH en ce Hen Hloc Hedges. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hhs Hbody].
    rewrite forallb_forall in Hhs.
    destruct Hedges as [Hhs_edges Hbody_edges].
    rewrite plain_rebuild_let_cont_rec, unbox_rebuild_let_cont_rec. cbv zeta.
    set (dss := map (fun '(k, h) => (k, handler_decisions sol unboxed false h))
                  handlers).
    set (masks := map (fun '(k, h) => (k, handler_mask sol false h)) handlers).
    assert (Hparams_of : ∀ (k : continuation) (h : continuation_handler),
      (k, h) ∈ handlers ->
      map is_kept_decision (handler_decisions sol unboxed false h)
      = handler_mask sol false h
      ∧ forallb is_decided (handler_decisions sol unboxed false h) = true).
    { intros k [params metadata handler] Hin.
      specialize (Hhs _ Hin). simpl in Hhs.
      apply andb_true_iff in Hhs as [Hhs _].
      apply andb_true_iff in Hhs as [_ Hparams].
      rewrite handler_decisions_plain by exact Hparams.
      split; [ | apply kept_decisions_decided ].
      apply kept_decisions_mask. unfold handler_mask.
      rewrite params_to_keep_length. reflexivity. }
    assert (Hmasks : map (fun '(k, ds) => (k, map is_kept_decision ds)) dss
                     = masks).
    { unfold dss, masks. rewrite map_map. apply map_ext_in.
      intros [k h] Hin. rewrite (proj1 (Hparams_of k h Hin)). reflexivity. }
    assert (Hen' : uinv (ubind_continuations en dss)).
    { apply uinv_continuations; [ exact Hen | ]. intros k ds Hin.
      unfold dss in Hin. apply in_map_iff in Hin as ([k' h] & Heq & Hin).
      injection Heq as <- <-. exact (proj2 (Hparams_of k' h Hin)). }
    assert (Henv : ue_env (ubind_continuations en dss)
                   = bind_continuations (ue_env en) masks).
    { cbn [ue_env ubind_continuations]. rewrite Hmasks. reflexivity. }
    destruct (IH _ _ Hen' Hbody Hbody_edges) as (IHub & IHfc & IHbs).
    rewrite Henv in IHub, IHfc, IHbs.
    assert (Hhandlers : ∀ hs : list (continuation * continuation_handler),
      (∀ (k : continuation) (h : continuation_handler),
         (k, h) ∈ hs -> (k, h) ∈ handlers) ->
      let hps :=
        map
          (fun '(k, h) =>
             (k,
              rebuild_handler F sol preserve_direct_calls unit_code_ids
                first_fresh (bind_continuations (ue_env en) masks) false
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
      ub_handlers unboxed hps hus
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
      destruct (IHh k h Hin (ubind_continuations en dss) _ Hen' (Hhs _ Hin)
                  (every_in _ _ (k, h) Hhs_edges Hin) false false)
        as (ps & md & hp & hu & Hp & Hu & Hps & Hhub & Hhfc & Hhbs).
      rewrite Henv in Hp. cbv zeta. cbn [map]. rewrite Hp, Hu.
      split; [ constructor; assumption | ].
      unfold handlers_bind_statics in *. cbn [existsb flat_map map fst].
      cbn [free_continuations_of_handler handler_binds_statics].
      rewrite Hhbs, Hbs_hs, Hhfc, Hfc_hs, Hfst_hs. split; [ | split ];
        reflexivity. }
    destruct (Hhandlers handlers (fun k h Hin => Hin))
      as (Hub_hs & Hbs_hs & Hfc_hs & Hfst_hs).
    split; [ constructor; assumption | ].
    unfold handlers_bind_statics in Hbs_hs.
    cbn [free_continuations_of_expr binds_statics].
    rewrite Hfst_hs, Hfc_hs, IHfc, IHbs, Hbs_hs. split; reflexivity.
  - (* Apply *)
    intros ap en ce Hen Hloc. discriminate Hloc.
  - (* Apply_cont *)
    intros ac en ce Hen Hloc _. unfold plain_rebuild, unbox_rebuild.
    cbn [rebuild_expr rebuild_expr_u]. simpl in Hloc.
    unfold rebuild_apply_cont, rebuild_apply_cont_u.
    rewrite (rewrite_apply_cont_same en ac Hen Hloc).
    destruct (rewrite_apply_cont sol (ue_env en) ac) as [ac' | ] eqn:Hac;
      [ | exact ub_result_invalid ].
    split; [ | split; reflexivity ].
    constructor. exact (rewrite_apply_cont_plain _ ac ac' Hloc Hac).
  - (* Switch *)
    intros scrutinee arms en ce Hen Hloc _. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hs Harms].
    unfold plain_rebuild, unbox_rebuild. cbn [rebuild_expr rebuild_expr_u].
    unfold rebuild_switch, rebuild_switch_u.
    assert (Hcompact : ∀ (d : Z) (ac : apply_cont),
               (d, ac) ∈ Keyed.compact Z.eqb arms ->
               plain_apply_cont unboxed ac = true).
    { intros d ac Hin.
      apply (keyed_in_compact Z.eqb z_eqb_decides_equality) in Hin.
      apply (keyed_lookup_in Z.eqb z_eqb_decides_equality) in Hin.
      rewrite forallb_forall in Harms. exact (Harms _ Hin). }
    assert (Harms' :
      ∀ l : list (Z * apply_cont),
        (∀ (d : Z) (ac : apply_cont), (d, ac) ∈ l ->
           plain_apply_cont unboxed ac = true) ->
        flat_map
          (fun '(d, ac) =>
             match rewrite_apply_cont_u sol unboxed en ac with
             | Some ac => [(d, ac)]
             | None => []
             end) l
        = flat_map
            (fun '(d, ac) =>
               match rewrite_apply_cont sol (ue_env en) ac with
               | Some ac => [(d, ac)]
               | None => []
               end) l
        ∧ forallb (fun '(_, ac) => plain_apply_cont unboxed ac)
            (flat_map
               (fun '(d, ac) =>
                  match rewrite_apply_cont sol (ue_env en) ac with
                  | Some ac => [(d, ac)]
                  | None => []
                  end) l)
          = true).
    { induction l as [ | [d ac] l IHl]; intros Hl; [ split; reflexivity | ].
      destruct IHl as [Heq Hplain].
      { intros d' ac' Hin. apply (Hl d' ac'). right. exact Hin. }
      simpl. rewrite Heq.
      rewrite (rewrite_apply_cont_same en ac Hen (Hl d ac (or_introl eq_refl))).
      destruct (rewrite_apply_cont sol (ue_env en) ac) as [ac' | ] eqn:Hac;
        simpl; [ | split; [ reflexivity | exact Hplain ] ].
      split; [ reflexivity | ]. rewrite Hplain, andb_true_r.
      exact (rewrite_apply_cont_plain _ ac ac' (Hl d ac (or_introl eq_refl))
               Hac). }
    destruct (Harms' _ Hcompact) as [Heq Hplain]. rewrite Heq.
    rewrite (rewrite_simple_u _ scrutinee Hs).
    destruct (Keyed.is_empty _); [ exact ub_result_invalid | ].
    split; [ | split; reflexivity ].
    constructor; [ apply rewrite_simple_plain, Hs | exact Hplain ].
  - (* Invalid *)
    intros en ce Hen Hloc _. exact ub_result_invalid.
  - (* Handler *)
    intros params metadata handler IH en ce Hen Hloc Hedges dead is_exn.
    simpl in Hloc.
    apply andb_true_iff in Hloc as [Hloc Hbody].
    apply andb_true_iff in Hloc as [Hlen Hparams]. apply Nat.eqb_eq in Hlen.
    pose proof (with_kinds_combine params (hm_params_kinds metadata) Hlen)
      as Hbound.
    pose proof (uinv_bind en (with_kinds params (hm_params_kinds metadata)) Hen)
      as Hen'.
    assert (Hbody' : local_expr unboxed sol
                       (env_kinds (ue_env (ubind_variables en
                          (with_kinds params (hm_params_kinds metadata)))))
                       handler = true).
    { cbn [ue_env ubind_variables ubind_variables_then ubind_set
           bind_variables_then env_kinds].
      rewrite Hbound. exact Hbody. }
    destruct (IH _ ce Hen' Hbody' Hedges) as (IHub & IHfc & IHbs).
    cbn [ue_env ubind_variables ubind_variables_then ubind_set] in IHub, IHfc,
      IHbs.
    rewrite plain_rebuild_handler, unbox_rebuild_handler. cbv zeta.
    unfold handler_mask.
    rewrite handler_decisions_plain by exact Hparams.
    rewrite params_of_kept_decisions
      by (rewrite params_to_keep_length; reflexivity).
    rewrite <- !filter_kept_map.
    rewrite Hbound, map_fst_combine, map_snd_combine by exact Hlen.
    rewrite <- Hbound.
    eexists _, _, _, _. split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ apply filter_kept_forallb, Hparams | ].
    destruct (dead && _); [ exact ub_result_invalid | ].
    split; [ exact IHub | split; assumption ].
Qed.

End Rebuilds.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Lemma uinv_empty : uinv empty_unbox_env.
Proof.
  split; [ reflexivity | reflexivity | reflexivity | ].
  intros k ds Hk. discriminate Hk.
Qed.

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
                   e [] Hloc))
    as Hreads.
  apply (proj1 (strong_local unboxed (solution_of_tables T)) e [] Hloc)
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
    ∀ (x : variable) (fields : unboxed_fields) (f : field) (v : variable),
      fields_of unboxed x = Some fields -> (f, Not_unboxed v) ∈ fields ->
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
              e empty_unbox_env (unit_conts u) uinv_empty Hloc' Hedges)
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
