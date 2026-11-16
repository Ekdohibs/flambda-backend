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

(* Proofs of the statements of FlambdaNormaliseSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaSemanticsProofs FlambdaNominalSpec
  FlambdaNominalProofs FlambdaControlFlowSpec FlambdaControlFlowProofs
  FlambdaContextSpec FlambdaContextProofs FlambdaApplySpec FlambdaApplyProofs
  FlambdaStaticSpec FlambdaStaticProofs FlambdaCompareSpec
  FlambdaTupledSpec FlambdaTupledProofs FlambdaPartialApplicationSpec
  FlambdaPartialApplicationProofs FlambdaNormaliseSpec.
From OxCamlFlambda2 Require FlambdaHeapProofs.
Import ListNotations.

(* ======================================================================== *)
(* Names in scope                                                           *)
(* ======================================================================== *)

Lemma simple_in_scope_sound (vs : list variable) (s : simple) (x : variable) :
  simple_in_scope vs s = true -> var_free_in_simple x s -> x ∈ vs.
Proof.
  destruct s as [y | sym | c]; simpl; try contradiction.
  intros Hin ->. apply among_In, Hin.
Qed.

Lemma simples_in_scope_sound (vs : list variable) (args : list simple)
    (x : variable) :
  simples_in_scope vs args = true -> var_free_in_simples x args -> x ∈ vs.
Proof.
  unfold simples_in_scope, var_free_in_simples. intros Hscope Hfree.
  apply Exists_exists in Hfree. destruct Hfree as (s & Hin & Hfree).
  rewrite forallb_forall in Hscope.
  exact (simple_in_scope_sound vs s x (Hscope s Hin) Hfree).
Qed.

Lemma value_slots_in_scope_sound (vs : list variable)
    (slots : list (value_slot * simple)) (x : variable) :
  forallb (fun slot => simple_in_scope vs (snd slot)) slots = true ->
  Exists (fun slot => var_free_in_simple x (snd slot)) slots ->
  x ∈ vs.
Proof.
  intros Hscope Hfree.
  apply Exists_exists in Hfree. destruct Hfree as (slot & Hin & Hfree).
  rewrite forallb_forall in Hscope.
  exact (simple_in_scope_sound vs (snd slot) x (Hscope slot Hin) Hfree).
Qed.

Lemma named_in_scope_sound (vs : list variable) (d : named) (x : variable) :
  named_in_scope vs d = true -> var_free_in_named x d -> x ∈ vs.
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure]; simpl.
  - apply simple_in_scope_sound.
  - apply simples_in_scope_sound.
  - apply value_slots_in_scope_sound.
  - apply value_slots_in_scope_sound.
  - apply simple_in_scope_sound.
  - apply simple_in_scope_sound.
  - apply simple_in_scope_sound.
Qed.

Lemma static_binding_in_scope_sound (vs : list variable)
    (binding : static_binding) (x : variable) :
  static_binding_in_scope vs binding = true ->
  var_free_in_static_binding x binding ->
  x ∈ vs.
Proof.
  destruct binding
    as [id c | id
       | sym [code slots | fslot syms decls slots | tag fields | s | bytes
              | elements]];
    simpl; try contradiction.
  - apply value_slots_in_scope_sound.
  - apply value_slots_in_scope_sound.
  - apply simples_in_scope_sound.
  - apply simple_in_scope_sound.
  - apply simples_in_scope_sound.
Qed.

Lemma all_among_In (names allowed : list positive) (k : positive) :
  FlambdaMapSpec.all_among names allowed = true -> k ∈ names -> k ∈ allowed.
Proof.
  unfold FlambdaMapSpec.all_among. rewrite forallb_forall. intros Hall Hin.
  apply among_In, Hall, Hin.
Qed.

Lemma no_exn_extra_args_nil (ap : apply_expr) :
  no_exn_extra_args ap = true -> ap_exn_extra_args ap = [].
Proof.
  unfold no_exn_extra_args.
  destruct (ap_exn_extra_args ap); [ reflexivity | discriminate ].
Qed.

(* What the check means: the free names of the expression are among those
   given. *)
Fixpoint expr_in_scope_sound (e : expr) :
  ∀ (vs : list variable) (ks : list continuation),
    expr_in_scope vs ks e = true ->
    (∀ x : variable, var_free_in_expr x e -> x ∈ vs)
    ∧ (∀ k : continuation, cont_free_in_expr k e -> k ∈ ks)
with handler_in_scope_sound (h : continuation_handler) :
  ∀ (vs : list variable) (ks : list continuation),
    handler_in_scope vs ks h = true ->
    (∀ x : variable, var_free_in_handler x h -> x ∈ vs)
    ∧ (∀ k : continuation, cont_free_in_handler k h -> k ∈ ks).
Proof.
  - destruct e
      as [ y kind d body | bindings body | k' h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros vs ks Hscope; simpl in Hscope.
    + (* Let *)
      apply andb_true_iff in Hscope as [Hd Hbody].
      destruct (expr_in_scope_sound body _ _ Hbody) as [Hvars Hconts].
      split; [ | exact Hconts ].
      intros x [Hx | [Hne Hx]];
        [ exact (named_in_scope_sound vs d x Hd Hx) | ].
      destruct (Hvars x Hx) as [Heq | Hin]; [ congruence | exact Hin ].
    + (* Let_static *)
      apply andb_true_iff in Hscope as [Hbindings Hbody].
      destruct (expr_in_scope_sound body _ _ Hbody) as [Hvars Hconts].
      split; [ | exact Hconts ].
      intros x [Hx | Hx]; [ | exact (Hvars x Hx) ].
      apply Exists_exists in Hx. destruct Hx as (binding & Hin & Hx).
      rewrite forallb_forall in Hbindings.
      exact (static_binding_in_scope_sound vs binding x
        (Hbindings binding Hin) Hx).
    + (* Let_cont_non_recursive *)
      apply andb_true_iff in Hscope as [Hh Hbody].
      destruct (handler_in_scope_sound h _ _ Hh) as [Hvars_h Hconts_h].
      destruct (expr_in_scope_sound body _ _ Hbody) as [Hvars Hconts].
      split.
      * intros x [Hx | Hx]; [ exact (Hvars_h x Hx) | exact (Hvars x Hx) ].
      * intros k [Hk | [Hne Hk]]; [ exact (Hconts_h k Hk) | ].
        destruct (Hconts k Hk) as [Heq | Hin]; [ congruence | exact Hin ].
    + (* Let_cont_recursive *)
      apply andb_true_iff in Hscope as [Hhandlers Hbody].
      destruct (expr_in_scope_sound body _ _ Hbody) as [Hvars Hconts].
      assert (Hin_handlers :
        (∀ x : variable,
          any_handler (var_free_in_handler x) handlers -> x ∈ vs)
        ∧ (∀ k : continuation,
            any_handler (cont_free_in_handler k) handlers ->
            k ∈ map fst handlers ++ ks)).
      { revert Hhandlers. generalize (map fst handlers ++ ks) as ks'.
        intros ks'. clear Hbody Hvars Hconts.
        induction handlers as [ | [k0 h] handlers IH]; simpl;
          [ intros _; split; intros ? [] | ].
        intros Hhandlers. apply andb_true_iff in Hhandlers as [Hh Hrest].
        destruct (handler_in_scope_sound h _ _ Hh) as [Hvars_h Hconts_h].
        destruct (IH Hrest) as [Hvars_rest Hconts_rest].
        split.
        - intros x [Hx | Hx];
            [ exact (Hvars_h x Hx) | exact (Hvars_rest x Hx) ].
        - intros k [Hk | Hk];
            [ exact (Hconts_h k Hk) | exact (Hconts_rest k Hk) ]. }
      destruct Hin_handlers as [Hvars_h Hconts_h].
      split.
      * intros x [Hx | Hx]; [ exact (Hvars_h x Hx) | exact (Hvars x Hx) ].
      * intros k [Hunbound Hk].
        assert (Hin : k ∈ map fst handlers ++ ks)
          by (destruct Hk as [Hk | Hk];
              [ exact (Hconts_h k Hk) | exact (Hconts k Hk) ]).
        apply in_app_or in Hin. destruct Hin as [Hin | Hin]; [ | exact Hin ].
        destruct (lookup_none_not_in k handlers Hunbound Hin).
    + (* Apply *)
      apply andb_true_iff in Hscope as [Hscope Hextra].
      apply andb_true_iff in Hscope as [Hscope Hexn].
      apply andb_true_iff in Hscope as [Hscope Hreturn].
      apply andb_true_iff in Hscope as [Hcallee Hargs].
      split.
      * intros x [Hx | [Hx | Hx]];
          [ exact (simple_in_scope_sound vs _ x Hcallee Hx)
          | exact (simples_in_scope_sound vs _ x Hargs Hx)
          | exact (simples_in_scope_sound vs _ x Hextra Hx) ].
      * intros k [-> | ->]; apply among_In; assumption.
    + (* Apply_cont *)
      unfold apply_cont_in_scope in Hscope.
      apply andb_true_iff in Hscope as [Hks Hargs].
      split.
      * intros x Hx. exact (simples_in_scope_sound vs _ x Hargs Hx).
      * intros k Hk. exact (all_among_In _ _ k Hks Hk).
    + (* Switch *)
      apply andb_true_iff in Hscope as [Hscrutinee Harms].
      rewrite forallb_forall in Harms.
      split.
      * intros x [Hx | Hx];
          [ exact (simple_in_scope_sound vs _ x Hscrutinee Hx) | ].
        apply Exists_exists in Hx. destruct Hx as (arm & Hin & Hx).
        specialize (Harms arm Hin). unfold apply_cont_in_scope in Harms.
        apply andb_true_iff in Harms as [_ Hargs].
        exact (simples_in_scope_sound vs _ x Hargs Hx).
      * intros k Hk. apply Exists_exists in Hk.
        destruct Hk as (arm & Hin & Hk).
        specialize (Harms arm Hin). unfold apply_cont_in_scope in Harms.
        apply andb_true_iff in Harms as [Hks _].
        exact (all_among_In _ _ k Hks Hk).
    + (* Invalid *)
      split; intros ? [].
  - destruct h as [params metadata handler]. intros vs ks Hscope.
    simpl in Hscope.
    destruct (expr_in_scope_sound handler _ _ Hscope) as [Hvars Hconts].
    split; [ | exact Hconts ].
    intros x [Hnotin Hx]. destruct (in_app_or _ _ _ (Hvars x Hx)); tauto.
Qed.

(* ======================================================================== *)
(* Inlining direct calls of known code                                      *)
(* ======================================================================== *)

Lemma var_in_simples_sound (x : variable) (args : list simple) :
  var_free_in_simples x args -> var_in_simples x args = true.
Proof.
  unfold var_free_in_simples, var_in_simples. intros Hfree.
  apply Exists_exists in Hfree. destruct Hfree as (s & Hin & Hfree).
  apply existsb_exists. exists s. split; [ exact Hin | ].
  destruct s as [y | sym | c]; simpl in Hfree; try contradiction.
  subst y. apply Pos.eqb_refl.
Qed.

(* [inline_renaming] sends the continuations of the code to those of the
   call, and fixes the variables. *)
Lemma inline_renaming_spec (ret exn k k_exn : continuation) :
  ret <> exn ->
  k <> k_exn ->
  (∀ x : variable, rename_variable (inline_renaming ret exn k k_exn) x = x)
  ∧ rename_continuation (inline_renaming ret exn k k_exn) ret = k
  ∧ rename_continuation (inline_renaming ret exn k k_exn) exn = k_exn.
Proof.
  intros Hcode Hcall. split; [ reflexivity | ].
  unfold rename_continuation, inline_renaming. simpl. split.
  - assert (Hk : swap ret k ret = k)
      by (unfold swap; rewrite Pos.eqb_refl; reflexivity).
    rewrite Hk. unfold swap at 1.
    destruct (Pos.eqb_spec k (swap ret k exn)) as [Heq | _].
    + rewrite <- Hk in Heq at 1. apply swap_injective in Heq.
      contradiction.
    + destruct (Pos.eqb_spec k k_exn); [ contradiction | reflexivity ].
  - unfold swap at 1. rewrite Pos.eqb_refl. reflexivity.
Qed.

Lemma inlinable_b_sound (ap : apply_expr) (c : code) :
  inlinable_b ap c = true ->
  inlinable ap c
    (inline_renaming (code_return_continuation c) (code_exn_continuation c)
       (ap_continuation ap) (ap_exn_continuation ap)).
Proof.
  unfold inlinable_b. intros Hb.
  apply andb_true_iff in Hb as [Hb Hstatic].
  apply negb_true_iff in Hstatic.
  apply andb_true_iff in Hb as [Hb Hfull].
  apply andb_true_iff in Hb as [Hb Harity].
  apply andb_true_iff in Hb as [Hb Hparams].
  apply andb_true_iff in Hb as [Hb Hmy_closure].
  apply andb_true_iff in Hb as [Hb Hextra].
  apply no_exn_extra_args_nil in Hextra.
  apply andb_true_iff in Hb as [Hb Hcall].
  apply andb_true_iff in Hb as [Hb Hcode].
  apply andb_true_iff in Hb as [Htupled Hscope].
  apply negb_true_iff in Htupled.
  apply negb_true_iff, Pos.eqb_neq in Hcode.
  apply negb_true_iff, Pos.eqb_neq in Hcall.
  destruct (expr_in_scope_sound _ _ _ Hscope) as [Hvars Hconts].
  destruct (inline_renaming_spec _ _ _ _ Hcode Hcall)
    as (Hfix & Hret & Hexn).
  unfold inlinable.
  refine
    (conj _
       (conj _
          (conj Hfix
             (conj Hret
                (conj Hexn
                   (conj Hextra
                      (conj _
                         (conj _
                            (conj _ (conj _ (conj Hstatic Htupled))))))))))).
  - intros x Hx. destruct (Hvars x Hx) as [<- | Hin]; auto.
  - intros k Hk. destruct (Hconts k Hk) as [<- | [<- | []]]; auto.
  - intros Hfree. apply var_in_simples_sound in Hfree.
    apply negb_true_iff in Hmy_closure. congruence.
  - intros param Hin Hfree. apply var_in_simples_sound in Hfree.
    rewrite forallb_forall in Hparams. specialize (Hparams param Hin).
    apply negb_true_iff in Hparams. congruence.
  - apply Nat.eqb_eq, Harity.
  - apply Nat.eqb_eq, Hfull.
Qed.

Lemma tupled_inlinable_b_sound (ap : apply_expr) (c : code)
    (tuple : simple) :
  tupled_inlinable_b ap c = true ->
  ap_args ap = [tuple] ->
  tupled_inlinable ap c
    (inline_renaming (code_return_continuation c) (code_exn_continuation c)
       (ap_continuation ap) (ap_exn_continuation ap))
    tuple.
Proof.
  unfold tupled_inlinable_b. intros Hb Hargs.
  apply andb_true_iff in Hb as [Hb Hstatic].
  apply negb_true_iff in Hstatic.
  apply andb_true_iff in Hb as [Hb Hnonempty].
  apply andb_true_iff in Hb as [Hb Hvalues].
  apply arity_eqb_eq in Hvalues.
  apply andb_true_iff in Hb as [Hb Harity].
  apply arity_eqb_eq in Harity.
  apply andb_true_iff in Hb as [Hb _].
  apply andb_true_iff in Hb as [Hb Hparams].
  apply andb_true_iff in Hb as [Hb Hmy_closure].
  apply andb_true_iff in Hb as [Hb Hextra].
  apply no_exn_extra_args_nil in Hextra.
  apply andb_true_iff in Hb as [Hb Hcall].
  apply andb_true_iff in Hb as [Hb Hcode].
  apply andb_true_iff in Hb as [Htupled Hscope].
  apply negb_true_iff, Pos.eqb_neq in Hcode.
  apply negb_true_iff, Pos.eqb_neq in Hcall.
  destruct (expr_in_scope_sound _ _ _ Hscope) as [Hvars Hconts].
  destruct (inline_renaming_spec _ _ _ _ Hcode Hcall)
    as (Hfix & Hret & Hexn).
  rewrite Hargs in Hmy_closure, Hparams.
  unfold tupled_inlinable.
  refine
    (conj _
       (conj _
          (conj Hfix
             (conj Hret
                (conj Hexn
                   (conj Hextra
                      (conj Hargs
                         (conj Harity
                            (conj _
                               (conj _
                                  (conj Htupled
                                     (conj Hvalues
                                        (conj _ Hstatic))))))))))))).
  - intros x Hx. destruct (Hvars x Hx) as [<- | Hin]; auto.
  - intros k Hk. destruct (Hconts k Hk) as [<- | [<- | []]]; auto.
  - intros Hfree.
    assert (Hin : var_free_in_simples (code_my_closure c) [tuple])
      by (apply Exists_cons_hd; exact Hfree).
    apply var_in_simples_sound in Hin.
    apply negb_true_iff in Hmy_closure. congruence.
  - intros param Hin Hfree.
    assert (Hin' : var_free_in_simples param [tuple])
      by (apply Exists_cons_hd; exact Hfree).
    apply var_in_simples_sound in Hin'.
    rewrite forallb_forall in Hparams. specialize (Hparams param Hin).
    apply negb_true_iff in Hparams. congruence.
  - intros Hnil. rewrite Hnil in Hnonempty. discriminate.
Qed.

(* The static environment binds the code that is known. *)
Definition statics_know (known : list (code_id * code))
    (statics : static_env) : Prop :=
  ∀ (id : code_id) (c : code),
    lookup id known = Some c -> lookup id (static_code statics) = Some c.

Lemma eval_static_bindings_know (vars : env)
    (bindings : list static_binding) :
  ∀ (known : list (code_id * code)) (statics extended : static_env),
    statics_know known statics ->
    eval_static_bindings vars bindings statics = Some extended ->
    statics_know (codes_of bindings known) extended.
Proof.
  induction bindings as [ | binding bindings IH];
    intros known statics extended Hknow; simpl.
  - intros [= <-]. exact Hknow.
  - destruct binding as [id c | id | sym const].
    + destruct (lookup id (static_code statics)); [ discriminate | ].
      apply IH. intros id' c'. simpl.
      destruct (Pos.eqb id' id); [ auto | apply Hknow ].
    + apply IH, Hknow.
    + destruct (lookup sym (static_values statics)); [ discriminate | ].
      destruct (eval_static_const vars const) as [allocated | ]; simpl;
        [ | discriminate ].
      apply IH. exact Hknow.
Qed.

(* A direct call whose callee is not a closure of the code called is
   undefined. *)
Lemma step_direct_not_closure (ap : apply_expr) (called : code_id)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record)
    (callee : value) :
  ap_call_kind ap = Direct called ->
  eval_simple vars (ap_callee ap) = Some callee ->
  (∀ slots : list (value_slot * value),
    closure_of statics callee <> Some (called, slots)) ->
  step (State (Apply ap) vars conts statics hp rd) = Undefined.
Proof.
  intros Hkind Hf Hother. rewrite step_apply, Hf, Hkind.
  destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | reflexivity ].
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
    [ | reflexivity ].
  unfold call_function, shape_of_call.
  replace
    (match partial_application_of callee with
     | Some _ | None =>
       shape_of_closure_call statics callee (Direct called) vs
         (ap_args_arity ap) (ap_return_arity ap)
     end)
    with
      (shape_of_closure_call statics callee (Direct called) vs
         (ap_args_arity ap) (ap_return_arity ap))
    by (destruct (partial_application_of callee); reflexivity).
  unfold shape_of_closure_call.
  destruct (closure_of statics callee) as [[called' slots] | ] eqn:Hclosure;
    [ | reflexivity ].
  destruct (Pos.eqb called' called) eqn:Heq; [ | reflexivity ].
  apply Pos.eqb_eq in Heq. subst called'. destruct (Hother slots eq_refl).
Qed.

Theorem split_over_application_refines :
  split_over_application_refines_statement.
Proof.
  intros called c ap vars conts statics hp rd Hkind Hcode Hcheck.
  unfold over_applied_b in Hcheck.
  apply andb_true_iff in Hcheck as [Hcheck Hextra].
  apply andb_true_iff in Hcheck as [Hcheck Hrest].
  apply andb_true_iff in Hcheck as [Hcheck Hexn].
  apply andb_true_iff in Hcheck as [Hcheck Hsize].
  apply andb_true_iff in Hcheck as [Htupled Hlonger].
  apply negb_true_iff in Htupled, Hexn, Hrest, Hextra.
  apply Pos.eqb_neq in Hexn. apply Nat.ltb_lt in Hlonger.
  apply Nat.leb_le in Hsize.
  (* the callee has no value, or is not a closure of the code called *)
  destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hf.
  2: { apply step_undefined_refines. rewrite step_apply, Hf. reflexivity. }
  assert (Hclosure :
    (∃ slots : list (value_slot * value),
      closure_of statics callee = Some (called, slots))
    ∨ ∀ slots : list (value_slot * value),
        closure_of statics callee <> Some (called, slots)).
  { destruct (closure_of statics callee) as [[called' slots] | ];
      [ | right; intros slots; discriminate ].
    destruct (Pos.eq_dec called' called) as [-> | Hother];
      [ left; exists slots; reflexivity | ].
    right. intros slots' [= Heq _]. exact (Hother Heq). }
  destruct Hclosure as [[slots Hclosure] | Hother].
  2: { apply step_undefined_refines.
       exact (step_direct_not_closure ap called vars conts statics hp rd callee
                Hkind Hf Hother). }
  destruct ap as [f kind args k' exn' extra args_arity return_arity].
  unfold exn_extra_simples in Hextra.
  cbn [ap_callee ap_call_kind ap_args ap_continuation ap_exn_continuation
       ap_exn_extra_args ap_args_arity ap_return_arity]
    in *.
  subst kind. unfold split_over_application.
  cbn [ap_callee ap_call_kind ap_args ap_continuation ap_exn_continuation
       ap_exn_extra_args ap_args_arity ap_return_arity].
  pose proof
    (over_application_split f (Direct called)
       (firstn (cardinal_unarized (code_params_arity c)) args)
       (skipn (cardinal_unarized (code_params_arity c)) args)
       (code_return_continuation c) k' exn' extra (code_my_closure c)
       (firstn (length (code_params_arity c)) args_arity)
       (skipn (length (code_params_arity c)) args_arity) return_arity vars
       conts statics hp rd callee called slots c Hf Hclosure Hcode Htupled)
    as Hsplit.
  rewrite !firstn_skipn in Hsplit. apply Hsplit.
  - apply firstn_length_le. lia.
  - apply firstn_length_le. exact Hsize.
  - intros Hnil. apply (f_equal (@length _)) in Hnil.
    rewrite length_skipn in Hnil. simpl in Hnil. lia.
  - exact Hexn.
  - intros Hfree. apply var_in_simples_sound in Hfree. congruence.
  - intros Hfree. apply var_in_simples_sound in Hfree. congruence.
Qed.

Section Kept.

(* The code whose calls are left alone. *)
Variable kept : list code_id.

Lemma inline_apply_refines (known : list (code_id * code)) (ap : apply_expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  statics_know known statics ->
  refines (State (Apply ap) vars conts statics hp rd)
    (State (inline_apply kept known ap) vars conts statics hp rd).
Proof.
  intros Hknow. unfold inline_apply.
  destruct (ap_call_kind ap) as [called | | code_ids] eqn:Hkind;
    [ | apply refines_refl | apply refines_refl ].
  destruct (lookup called known) as [c | ] eqn:Hlookup;
    [ | apply refines_refl ].
  destruct (negb (among called kept) && inlinable_b ap c) eqn:Hcheck.
  - apply andb_true_iff in Hcheck as [_ Hinlinable].
    exact (inline ap called c _ vars conts statics _ rd_ Hkind
      (Hknow called c Hlookup) (inlinable_b_sound ap c Hinlinable)).
  - (* tupled code, given its tuple *)
    destruct (negb (among called kept) && tupled_inlinable_b ap c)
      eqn:Htupled.
    + apply andb_true_iff in Htupled as [_ Htupled].
      unfold inlined_tupled_call.
      destruct (ap_args ap) as [ | tuple [ | arg args]] eqn:Hargs;
        try apply refines_refl.
      exact (inline_tupled ap called c _ tuple vars conts statics _ rd_ Hkind
        (Hknow called c Hlookup)
        (tupled_inlinable_b_sound ap c tuple Htupled Hargs)).
    + (* more arguments than the code has parameters: the call split *)
      destruct (over_applied_b ap c) eqn:Hover; [ | apply refines_refl ].
      exact (split_over_application_refines called c ap vars conts statics
               hp rd Hkind (Hknow called c Hlookup) Hover).
Qed.

Lemma statics_know_extend (known : list (code_id * code))
    (statics statics' : static_env) :
  statics_know known statics -> statics_extend statics statics' ->
  statics_know known statics'.
Proof. intros Hknow [Hcode _] id c Hid. apply Hcode, Hknow, Hid. Qed.

(* Handlers that refine each other's counterparts where they run, in the
   two forms that FlambdaContextProofs.related_envs_refines asks for. *)
Definition handlers_refine (ok : static_env -> Prop) (vars : env)
    (handlers handlers' : list (continuation * continuation_handler))
    : Prop :=
  handlers_related (handler_refines_where ok) vars handlers handlers'
  ∧ handlers_related
      (fun params kinds def_vars target source =>
         handler_refines_where ok params kinds def_vars source target)
      vars handlers' handlers.

(* Inlining leaves what binds something static as it is: what is inlined
   binds nothing. *)
Lemma let_params_binds_statics (params : list variable) :
  ∀ (kinds : list flambda_kind) (args : list simple) (body : expr),
    binds_statics (let_params params kinds args body) = binds_statics body.
Proof.
  induction params as [ | param params IH]; intros kinds [ | arg args] body;
    simpl; auto.
Qed.

Lemma inlined_body_binds_statics (c : code) (r : renaming) (ap : apply_expr) :
  binds_statics (inlined_body c r ap) = binds_statics (code_body c).
Proof.
  unfold inlined_body. simpl. rewrite let_params_binds_statics.
  apply rename_binds_statics.
Qed.

Lemma inlinable_b_binds_statics (ap : apply_expr) (c : code) :
  inlinable_b ap c = true -> binds_statics (code_body c) = false.
Proof.
  unfold inlinable_b. intros Hb. apply andb_true_iff in Hb as [_ Hstatic].
  apply negb_true_iff, Hstatic.
Qed.

Lemma let_tuple_fields_binds_statics (params : list variable) :
  ∀ (size field : nat) (tuple : simple) (body : expr),
    binds_statics (let_tuple_fields params size field tuple body)
    = binds_statics body.
Proof.
  induction params as [ | param params IH]; intros size field tuple body;
    simpl; auto.
Qed.

Lemma inlined_tupled_call_binds_statics (c : code) (r : renaming)
    (ap : apply_expr) :
  binds_statics (code_body c) = false ->
  binds_statics (inlined_tupled_call c r ap) = false.
Proof.
  intros Hbody. unfold inlined_tupled_call.
  destruct (ap_args ap) as [ | tuple [ | arg args]]; try reflexivity.
  unfold inlined_tupled_body. simpl.
  rewrite let_tuple_fields_binds_statics, rename_binds_statics.
  exact Hbody.
Qed.

Lemma tupled_inlinable_b_binds_statics (ap : apply_expr) (c : code) :
  tupled_inlinable_b ap c = true -> binds_statics (code_body c) = false.
Proof.
  unfold tupled_inlinable_b. intros Hb.
  apply andb_true_iff in Hb as [_ Hstatic]. apply negb_true_iff, Hstatic.
Qed.

Lemma inline_apply_binds_statics (known : list (code_id * code))
    (ap : apply_expr) :
  binds_statics (inline_apply kept known ap) = false.
Proof.
  unfold inline_apply.
  destruct (ap_call_kind ap) as [called | | code_ids];
    [ | reflexivity | reflexivity ].
  destruct (lookup called known) as [c | ]; [ | reflexivity ].
  destruct (negb (among called kept) && inlinable_b ap c) eqn:Hcheck.
  - apply andb_true_iff in Hcheck as [_ Hinlinable].
    rewrite inlined_body_binds_statics.
    exact (inlinable_b_binds_statics ap c Hinlinable).
  - destruct (negb (among called kept) && tupled_inlinable_b ap c)
      eqn:Htupled;
      [ | destruct (over_applied_b ap c); reflexivity ].
    apply andb_true_iff in Htupled as [_ Htupled].
    apply inlined_tupled_call_binds_statics.
    exact (tupled_inlinable_b_binds_statics ap c Htupled).
Qed.

Fixpoint inline_calls_binds_statics (e : expr) :
  ∀ known : list (code_id * code),
    binds_statics (inline_calls kept known e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros known; simpl; try reflexivity.
  - apply inline_calls_binds_statics.
  - rewrite inline_calls_binds_statics. f_equal.
    induction bindings as [ | [id c | id | sym const] bindings IH]; simpl;
      congruence.
  - destruct h as [params metadata handler]. simpl.
    rewrite !inline_calls_binds_statics. reflexivity.
  - rewrite inline_calls_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite inline_calls_binds_statics, IH. reflexivity.
  - apply inline_apply_binds_statics.
Qed.

Lemma inline_calls_handlers_bind_statics (known : list (code_id * code))
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map (fun '(k, h) => (k, inline_calls_in_handler kept known h)) handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
    [ reflexivity | ].
  rewrite inline_calls_binds_statics, IH. reflexivity.
Qed.

(* Groups of bindings whose code has bodies that refine their
   counterparts, in the two forms that
   FlambdaContextProofs.related_states_refines asks for. *)
Definition bindings_refine (ok : static_env -> Prop)
    (bindings bindings' : list static_binding) : Prop :=
  Forall2 (binding_related (expr_refines_where ok)) bindings bindings'
  ∧ Forall2
      (binding_related
         (fun target source => expr_refines_where ok source target))
      bindings' bindings.

(* Down the term, with the static environment binding what is known: at
   once in a body, and from then on in a handler. *)
Fixpoint inline_calls_refines_knowing (e : expr) :
  ∀ (known : list (code_id * code)) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    statics_know known statics ->
    refines (State e vars conts statics hp rd)
      (State (inline_calls kept known e) vars conts statics hp rd).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body
         | ap | ac | scrutinee arms | ];
      intros known vars conts statics hp rd Hknow; simpl;
      try apply refines_refl.
    + apply let_refines_step_exec; [ reflexivity | intros v hp' rd' _ _ ].
      apply inline_calls_refines_knowing, Hknow.
    + (* the code of the group is rewritten: the body under the one
         static environment refines it under the other; then down the
         body, the static environment binding the code as rewritten *)
      assert (Hrefine :
        bindings_refine (statics_know known) bindings
          (map (inline_calls_in_binding kept known) bindings)).
      { induction bindings
          as [ | [id [params my_closure ret exn metadata code_body] | id
                 | sym const]
                 bindings IH];
          simpl.
        - split; apply Forall2_nil.
        - destruct IH as [Hforwards Hbackwards].
          split; (apply Forall2_cons; [ | assumption ]);
            apply Binding_code, Codes_related;
            first
              [ apply inline_calls_binds_statics
              | symmetry; apply inline_calls_binds_statics
              | intros vars' conts' statics' hp' rd' Hknow';
                apply inline_calls_refines_knowing, Hknow' ].
        - destruct IH as [Hforwards Hbackwards].
          split; (apply Forall2_cons; [ apply Binding_same | assumption ]).
        - destruct IH as [Hforwards Hbackwards].
          split; (apply Forall2_cons; [ apply Binding_same | assumption ]). }
      remember (map (inline_calls_in_binding kept known) bindings) as bindings'
        eqn:Hdef.
      clear Hdef. destruct Hrefine as [Hforwards Hbackwards].
      apply (fun H =>
        eval_static_bindings_related _ vars _ _ H statics statics
          (statics_related_refl _ statics)) in Hforwards, Hbackwards.
      apply let_static_refines_step.
      destruct (eval_static_bindings vars bindings statics) as [extended | ]
        eqn:Hbindings,
        (eval_static_bindings vars bindings' statics) as [extended' | ]
          eqn:Hbindings';
        try contradiction; [ | exact I ].
      apply (refines_trans _ (State body vars conts extended' hp rd)).
      * apply (related_states_refines (statics_know known)
          (statics_know_extend known));
          [ exact (statics_know_extend known _ _ Hknow
              (eval_static_bindings_extend vars _ _ _ Hbindings))
          | exact (statics_know_extend known _ _ Hknow
              (eval_static_bindings_extend vars _ _ _ Hbindings'))
          | apply cont_env_related_refl | apply cont_env_related_refl
          | exact Hforwards | exact Hbackwards ].
      * apply inline_calls_refines_knowing.
        exact (eval_static_bindings_know vars bindings' known statics
          extended' Hknow Hbindings').
    + destruct h as [params metadata handler]. simpl.
      apply (refines_trans _
        (State
           (Let_cont_non_recursive k
              (Handler params metadata (inline_calls kept known handler)) body)
           vars conts statics hp rd)).
      * apply (let_cont_handler_refines_where (statics_know known)
          (statics_know_extend known)); [ exact Hknow | ].
        intros vs handler_vars conts' statics' hp' rd' Hknow' _ _.
        apply inline_calls_refines_knowing, Hknow'.
      * apply let_cont_non_recursive_refines_step,
          inline_calls_refines_knowing, Hknow.
    + apply (refines_trans _
        (State
           (Let_cont_recursive
              (map (fun '(k, h) => (k, inline_calls_in_handler kept known h))
                 handlers)
              body)
           vars conts statics hp rd)).
      * assert (Hhandlers :
          handlers_refine (statics_know known) vars handlers
            (map (fun '(k, h) => (k, inline_calls_in_handler kept known h))
               handlers)).
        { induction handlers
            as [ | [k [params metadata handler]] handlers IH];
            simpl.
          - split; apply Forall2_nil.
          - destruct IH as [Hforwards Hbackwards].
            split; (apply Forall2_cons; [ | assumption ]);
              (split; [ reflexivity | ]); apply Handler_related;
              intros vs handler_vars conts' statics' hp' rd' Hknow' _ _;
              apply inline_calls_refines_knowing, Hknow'. }
        destruct Hhandlers as [Hforwards Hbackwards].
        apply (let_cont_recursive_handlers_refine_where (statics_know known)
          (statics_know_extend known));
          try assumption.
        symmetry. apply inline_calls_handlers_bind_statics.
      * apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
        apply inline_calls_refines_knowing, Hknow.
    + apply inline_apply_refines, Hknow.
Qed.

End Kept.

Theorem inline_calls_refines : inline_calls_refines_statement.
Proof.
  intros kept e vars conts statics hp rd. apply inline_calls_refines_knowing.
  intros id c Hlookup. discriminate Hlookup.
Qed.

(* ======================================================================== *)
(* Removing unused [Let]s                                                   *)
(* ======================================================================== *)

Lemma var_in_simple_sound (x : variable) (s : simple) :
  var_free_in_simple x s -> var_in_simple x s = true.
Proof.
  destruct s as [y | sym | c]; simpl; try contradiction.
  intros ->. apply Pos.eqb_refl.
Qed.

Lemma var_in_value_slots_sound (x : variable)
    (slots : list (value_slot * simple)) :
  Exists (fun slot => var_free_in_simple x (snd slot)) slots ->
  var_in_value_slots x slots = true.
Proof.
  intros Hfree. apply Exists_exists in Hfree.
  destruct Hfree as (slot & Hin & Hfree). apply existsb_exists.
  exists slot. split; [ exact Hin | apply var_in_simple_sound, Hfree ].
Qed.

Lemma var_in_named_sound (x : variable) (d : named) :
  var_free_in_named x d -> var_in_named x d = true.
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure]; simpl.
  - apply var_in_simple_sound.
  - apply var_in_simples_sound.
  - apply var_in_value_slots_sound.
  - apply var_in_value_slots_sound.
  - apply var_in_simple_sound.
  - apply var_in_simple_sound.
  - apply var_in_simple_sound.
Qed.

Lemma var_in_static_binding_sound (x : variable) (binding : static_binding) :
  var_free_in_static_binding x binding ->
  var_in_static_binding x binding = true.
Proof.
  destruct binding
    as [id c | id
       | sym [code slots | fslot syms decls slots | tag fields | s | bytes
              | elements]];
    simpl; try contradiction.
  - apply var_in_value_slots_sound.
  - apply var_in_value_slots_sound.
  - apply var_in_simples_sound.
  - apply var_in_simple_sound.
  - apply var_in_simples_sound.
Qed.

(* The test says yes of every variable that is free. *)
Fixpoint var_free_b_sound (x : variable) (e : expr) {struct e} :
  var_free_in_expr x e -> var_free_b x e = true
with var_free_in_handler_b_sound (x : variable) (h : continuation_handler)
    {struct h} :
  var_free_in_handler x h -> var_free_in_handler_b x h = true.
Proof.
  - destruct e
      as [ y kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl.
    + intros [Hx | [Hne Hx]]; apply orb_true_iff;
        [ left; apply var_in_named_sound, Hx | right ].
      apply andb_true_iff. split.
      * apply negb_true_iff, Pos.eqb_neq, Hne.
      * apply (var_free_b_sound x body), Hx.
    + intros [Hx | Hx]; apply orb_true_iff;
        [ left | right; apply (var_free_b_sound x body), Hx ].
      apply Exists_exists in Hx. destruct Hx as (binding & Hin & Hx).
      apply existsb_exists. exists binding.
      split; [ exact Hin | apply var_in_static_binding_sound, Hx ].
    + intros [Hx | Hx]; apply orb_true_iff;
        [ left; apply (var_free_in_handler_b_sound x h), Hx
        | right; apply (var_free_b_sound x body), Hx ].
    + intros [Hx | Hx]; apply orb_true_iff;
        [ left | right; apply (var_free_b_sound x body), Hx ].
      induction handlers as [ | [k h] handlers IH]; simpl in Hx |- *;
        [ contradiction | ].
      apply orb_true_iff. destruct Hx as [Hx | Hx];
        [ left; apply (var_free_in_handler_b_sound x h), Hx
        | right; exact (IH Hx) ].
    + intros [Hx | [Hx | Hx]]; rewrite !orb_true_iff;
        [ left; left; apply var_in_simple_sound, Hx
        | left; right; apply var_in_simples_sound, Hx
        | right; apply var_in_simples_sound, Hx ].
    + apply var_in_simples_sound.
    + intros [Hx | Hx]; apply orb_true_iff;
        [ left; apply var_in_simple_sound, Hx | right ].
      apply Exists_exists in Hx. destruct Hx as (arm & Hin & Hx).
      apply existsb_exists. exists arm.
      split; [ exact Hin | apply var_in_simples_sound, Hx ].
    + contradiction.
  - destruct h as [params metadata handler]. simpl. intros [Hnotin Hx].
    apply andb_true_iff. split.
    + apply negb_true_iff. destruct (among x params) eqn:Hin;
        [ | reflexivity ].
      apply among_In in Hin. contradiction.
    + apply (var_free_b_sound x handler), Hx.
Qed.

Lemma expr_refines_trans (e e' e'' : expr) :
  expr_refines e e' -> expr_refines e' e'' -> expr_refines e e''.
Proof.
  intros H H' vars conts statics hp rd.
  exact
    (refines_trans _ _ _ (H vars conts statics hp rd_)
       (H' vars conts statics hp rd_)).
Qed.

Lemma expr_refines_refl (e : expr) : expr_refines e e.
Proof. intros vars conts statics hp rd. apply refines_refl. Qed.

(* Each layer is a context around what the induction gives; a [Let] whose
   variable the test finds unused then goes. *)
Fixpoint remove_unused_lets_refines_aux (e : expr) :
  expr_refines e (remove_unused_lets e)
with remove_unused_lets_in_handler_refines (h : continuation_handler) :
  ∀ (k : continuation) (body : expr),
    expr_refines (Let_cont_non_recursive k h body)
      (Let_cont_non_recursive k (remove_unused_lets_in_handler h) body).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try apply expr_refines_refl.
    + pose proof (plug_refines (Let_body x kind d Hole) _ _ (not_many _)
        (remove_unused_lets_refines_aux body)) as Hbody.
      simpl in Hbody.
      destruct
        (var_free_b x (remove_unused_lets body) || negb (named_may_go d))
        eqn:Hkeep; [ exact Hbody | ].
      apply orb_false_iff in Hkeep as [Hfree Hgo].
      apply negb_false_iff in Hgo.
      apply (expr_refines_trans _ _ _ Hbody), unused_let.
      * destruct d as [s | [p | p | p | q | rp] args | | | | | ];
          try reflexivity.
        discriminate Hgo.
      * intros Hx. apply var_free_b_sound in Hx. congruence.
    + exact (plug_refines (Let_static_body bindings Hole) _ _ (not_many _)
        (remove_unused_lets_refines_aux body)).
    + apply (expr_refines_trans _ _ _
        (remove_unused_lets_in_handler_refines h k body)).
      exact (plug_refines
        (Let_cont_non_recursive_body k (remove_unused_lets_in_handler h)
           Hole)
        _ _ (not_many _) (remove_unused_lets_refines_aux body)).
    + exact (plug_refines (Let_cont_recursive_body handlers Hole)
          _ _ (not_many _)
        (remove_unused_lets_refines_aux body)).
  - destruct h as [params metadata handler]. intros k body. simpl.
    exact (plug_refines
      (Let_cont_non_recursive_handler k params metadata Hole body) _ _
      (not_many _) (remove_unused_lets_refines_aux handler)).
Qed.

Theorem remove_unused_lets_refines : remove_unused_lets_refines_statement.
Proof. exact remove_unused_lets_refines_aux. Qed.

(* ======================================================================== *)
(* A continuation applied once, at the end of its scope                     *)
(* ======================================================================== *)

Lemma bind_params_defined (params : list variable) :
  ∀ (vs : list value) (vars vars' bound : env),
    bind_params params vs vars = Some bound ->
    ∃ bound' : env, bind_params params vs vars' = Some bound'.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] vars vars' bound;
    simpl; try discriminate; [ eauto | ].
  apply IH.
Qed.

(* Binding the same parameters under environments that agree at a variable
   gives environments that agree at it; and at every parameter. *)
Lemma bind_params_agree (params : list variable) :
  ∀ (vs : list value) (vars vars' bound bound' : env),
    bind_params params vs vars = Some bound ->
    bind_params params vs vars' = Some bound' ->
    ∀ x : variable,
      x ∈ params ∨ lookup x vars' = lookup x vars ->
      lookup x bound' = lookup x bound.
Proof.
  induction params as [ | p params IH];
    intros [ | v vs] vars vars' bound bound'; simpl; try discriminate.
  - intros [= <-] [= <-] x [[] | Hagree]. exact Hagree.
  - intros Hbind Hbind' x Hx. apply (IH vs _ _ _ _ Hbind Hbind'). simpl.
    destruct (Pos.eqb_spec x p) as [-> | Hne]; [ right; reflexivity | ].
    destruct Hx as [[Heq | Hin] | Hagree];
      [ congruence | left; exact Hin | right; exact Hagree ].
Qed.

(* The body under the binding of [k], against the body with the handler in
   place: the [Let]s run alike; then the one applies [k] and the other
   binds the parameters and goes on, to the handler under environments
   that agree on its free variables. *)
Lemma inline_cont_at_tail_refines (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler : expr) (body : expr) :
  ∀ inlined : expr,
    inline_cont_at_tail k params metadata handler body = Some inlined ->
    ∀ (vars def_vars : env) (conts : cont_env) (statics : static_env)
      (hp : heap) (rd : read_record),
      (∀ x : variable,
        var_free_in_handler x (Handler params metadata handler) ->
        lookup x vars = lookup x def_vars) ->
      refines
        (State body vars
           ((k, Closure (Handler params metadata handler) def_vars conts [])
            :: conts)
           statics hp rd)
        (State inlined vars conts statics hp rd).
Proof.
  induction body
    as [ x kind d body IH | bindings body IH | k' h body IH | handlers body IH
       | ap | ac | scrutinee arms | ];
    intros inlined Heq vars def_vars conts statics hp rd Hagree; simpl in Heq;
    try discriminate.
  - (* Let *)
    match type of Heq with
    | context [if ?free then _ else _] => destruct free eqn:Hfree
    end;
      [ discriminate | ].
    change (var_free_in_handler_b x (Handler params metadata handler) = false)
      in Hfree.
    destruct (inline_cont_at_tail k params metadata handler body)
      as [body' | ];
      [ | discriminate ].
    simpl in Heq. injection Heq as <-.
    apply let_refines_step_exec; [ reflexivity | intros v hp' rd' _ _ ].
    apply (IH body' eq_refl). intros y Hy. simpl.
    destruct (Pos.eqb_spec y x) as [-> | _]; [ | exact (Hagree y Hy) ].
    apply var_free_in_handler_b_sound in Hy. congruence.
  - (* Let_static *)
    destruct (inline_cont_at_tail k params metadata handler body)
      as [body' | ];
      [ | discriminate ].
    simpl in Heq. injection Heq as <-.
    apply let_static_refines_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ];
      [ | exact I ].
    exact (IH body' eq_refl vars def_vars conts extended hp rd_ Hagree).
  - (* Apply_cont *)
    destruct
      (Pos.eqb (ac_continuation ac) k
       && no_trap_action ac
       && Nat.eqb (length params) (length (ac_args ac))
       && forallb (fun param => negb (var_in_simples param (ac_args ac)))
            params)
      eqn:Hcheck; [ | discriminate ].
    injection Heq as <-.
    apply andb_true_iff in Hcheck as [Hcheck Hcapture].
    apply andb_true_iff in Hcheck as [Hcheck _].
    apply andb_true_iff in Hcheck as [Hk _]. apply Pos.eqb_eq in Hk.
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs;
      [ | apply step_undefined_refines; rewrite step_apply_cont, Hargs;
          reflexivity ].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
      eqn:Hkinds;
      [ | apply step_undefined_refines; rewrite step_apply_cont, Hargs, Hk;
          unfold apply_continuation; rewrite resolve_head; simpl;
          rewrite Hkinds; reflexivity ].
    destruct (bind_params params vs def_vars) as [handler_vars | ] eqn:Hbind;
      [ | apply step_undefined_refines; rewrite step_apply_cont, Hargs, Hk;
          unfold apply_continuation; rewrite resolve_head; simpl;
          rewrite Hkinds, Hbind; reflexivity ].
    destruct (bind_params_defined params vs def_vars vars handler_vars Hbind)
      as [bound Hbound].
    destruct (let_params_run params (hm_params_kinds metadata) (ac_args ac)
                vs vars bound handler conts statics hp rd Hargs
                (proj1 (kinds_eqb_eq _ _) Hkinds))
      as [n Hrun]; [ | exact Hbound | ].
    { intros param Hin Hfree. apply var_in_simples_sound in Hfree.
      rewrite forallb_forall in Hcapture. specialize (Hcapture param Hin).
      apply negb_true_iff in Hcapture. congruence. }
    apply equivalent_refines.
    apply (equivalent_trans _ (State handler handler_vars conts statics hp rd)).
    { apply step_equivalent.
      apply (step_apply_cont_handler _ _ _ _ _ rd_ vs params metadata handler
        def_vars conts []);
        [ exact Hargs | rewrite Hk; apply resolve_head | exact Hkinds
        | exact Hbind ]. }
    apply (equivalent_trans _ (State handler bound conts statics hp rd)).
    { apply coincide_equivalent.
      - intros x Hx.
        apply (bind_params_agree params vs def_vars vars handler_vars bound
          Hbind Hbound).
        destruct (in_dec Pos.eq_dec x params) as [Hin | Hnotin];
          [ left; exact Hin | right ].
        apply Hagree. split; assumption.
      - intros k' _. apply (conts_agree_same identity_renaming). reflexivity. }
    apply equivalent_sym, (equivalent_of_run n 0). exact Hrun.
Qed.

(* Each layer is a context around what the induction gives; then the
   handler goes into the body where that ends by applying it. *)
Fixpoint inline_tail_conts_refines_aux (e : expr) :
  expr_refines e (inline_tail_conts e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try apply expr_refines_refl.
  - exact (plug_refines (Let_body x kind d Hole) _ _ (not_many _)
      (inline_tail_conts_refines_aux body)).
  - exact (plug_refines (Let_static_body bindings Hole) _ _ (not_many _)
      (inline_tail_conts_refines_aux body)).
  - assert (Hlayers :
      expr_refines
        (Let_cont_non_recursive k (Handler params metadata handler) body)
        (Let_cont_non_recursive k
           (Handler params metadata (inline_tail_conts handler))
           (inline_tail_conts body))).
    { apply (expr_refines_trans _
        (Let_cont_non_recursive k
           (Handler params metadata (inline_tail_conts handler)) body)).
      - exact (plug_refines
          (Let_cont_non_recursive_handler k params metadata Hole body)
          _ _ (not_many _)
          (inline_tail_conts_refines_aux handler)).
      - exact (plug_refines
          (Let_cont_non_recursive_body k
             (Handler params metadata (inline_tail_conts handler)) Hole)
          _ _ (not_many _) (inline_tail_conts_refines_aux body)). }
    destruct
      (inline_cont_at_tail k params metadata (inline_tail_conts handler)
         (inline_tail_conts body))
      as [inlined | ] eqn:Hinline; [ | exact Hlayers ].
    apply (expr_refines_trans _ _ _ Hlayers). intros vars conts statics hp rd.
    eapply refines_trans;
      [ apply equivalent_refines, step_equivalent,
          step_let_cont_non_recursive
      | ].
    apply (inline_cont_at_tail_refines k params metadata _ _ inlined
      Hinline).
    intros x _. reflexivity.
  - exact (plug_refines (Let_cont_recursive_body handlers Hole) _ _ (not_many _)
      (inline_tail_conts_refines_aux body)).
Qed.

Theorem inline_tail_conts_refines : inline_tail_conts_refines_statement.
Proof. exact inline_tail_conts_refines_aux. Qed.

(* ======================================================================== *)
(* Bindings of deleted code                                                 *)
(* ======================================================================== *)

Lemma expr_equivalent_trans (e e' e'' : expr) :
  expr_equivalent e e' -> expr_equivalent e' e'' -> expr_equivalent e e''.
Proof.
  intros H H' vars conts statics hp rd.
  exact (equivalent_trans _ _ _ (H vars conts statics hp rd_)
    (H' vars conts statics hp rd_)).
Qed.

Lemma eval_static_bindings_filter (vars : env)
    (bindings : list static_binding) :
  ∀ statics : static_env,
    eval_static_bindings vars (filter binds_something bindings) statics
    = eval_static_bindings vars bindings statics.
Proof.
  induction bindings as [ | binding bindings IH]; intros statics;
    [ reflexivity | ].
  destruct binding as [id c | id | sym const]; simpl.
  - destruct (lookup id (static_code statics)); [ reflexivity | apply IH ].
  - apply IH.
  - destruct (lookup sym (static_values statics)); [ reflexivity | ].
    destruct (eval_static_const vars const); simpl;
      [ apply IH | reflexivity ].
Qed.

(* A group and the group without its bindings of deleted code, the
   [Let_static] itself going if nothing is left. *)
Lemma let_static_if_any_equivalent (bindings : list static_binding)
    (body : expr) :
  expr_equivalent (Let_static bindings body)
    (let_static_if_any (filter binds_something bindings) body).
Proof.
  intros vars conts statics hp rd.
  destruct (filter binds_something bindings) as [ | binding kept] eqn:Hkept;
    simpl.
  - apply step_equivalent.
    rewrite step_let_static,
      <- (eval_static_bindings_filter vars bindings statics), Hkept.
    reflexivity.
  - apply step_eq_equivalent.
    rewrite !step_let_static,
      <- (eval_static_bindings_filter vars bindings statics), Hkept.
    reflexivity.
Qed.

Fixpoint drop_deleted_code_equivalent_aux (e : expr) :
  expr_equivalent e (drop_deleted_code e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try (intros vars conts statics hp rd; apply equivalent_refl).
  - exact (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _)
      (drop_deleted_code_equivalent_aux body)).
  - apply (expr_equivalent_trans _
      (Let_static bindings (drop_deleted_code body))).
    + exact (plug_equivalent (Let_static_body bindings Hole) _ _ (not_many _)
        (drop_deleted_code_equivalent_aux body)).
    + apply let_static_if_any_equivalent.
  - apply (expr_equivalent_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (drop_deleted_code handler)) body)).
    + exact (plug_equivalent
        (Let_cont_non_recursive_handler k params metadata Hole body)
        _ _ (not_many _)
        (drop_deleted_code_equivalent_aux handler)).
    + exact (plug_equivalent
        (Let_cont_non_recursive_body k
           (Handler params metadata (drop_deleted_code handler)) Hole)
        _ _ (not_many _) (drop_deleted_code_equivalent_aux body)).
  - exact (plug_equivalent (Let_cont_recursive_body handlers Hole)
        _ _ (not_many _)
      (drop_deleted_code_equivalent_aux body)).
Qed.

Theorem drop_deleted_code_equivalent : drop_deleted_code_equivalent_statement.
Proof. exact drop_deleted_code_equivalent_aux. Qed.

(* ======================================================================== *)
(* Folding constants                                                        *)
(* ======================================================================== *)

Lemma consts_of_simples_eval (vars : env) (args : list simple) :
  ∀ cs : list const,
    consts_of_simples args = Some cs ->
    eval_simples vars args = Some (map Value_const cs).
Proof.
  induction args as [ | [x | sym | c] args IH]; intros cs; simpl;
    try discriminate.
  - intros [= <-]. reflexivity.
  - destruct (consts_of_simples args) as [cs' | ]; [ | discriminate ].
    simpl. intros [= <-]. rewrite (IH cs' eq_refl). reflexivity.
Qed.

(* What [fold_named] makes of a primitive: one of its two cases that
   fold no constant, or the folding. *)
Lemma fold_named_prim_cases (p : prim) (args : list simple) :
  (∃ (kind : array_kind) (set_kind : array_set_kind),
     p = Heap (Array_set kind set_kind)
     ∧ fold_named (Prim p args)
       = Prim (Heap (Array_set (array_kind_of_set_kind set_kind) set_kind))
           args)
  ∨ (∃ kind : array_kind_for_length,
       p = Unary (Array_length kind)
       ∧ fold_named (Prim p args)
         = Prim (Unary (Array_length (Array_kind Array_values))) args)
  ∨ fold_named (Prim p args)
    = match (if prim_may_fold p then consts_of_simples args else None) with
      | Some cs =>
        match eval_prim empty_static_env p (map Value_const cs) with
        | Some (Value_const c) => Simple (Const c)
        | _ => Prim p args
        end
      | None => Prim p args
      end.
Proof.
  destruct p as [u | b | v | h | r]; try (right; right; reflexivity).
  - destruct u; try (right; right; reflexivity).
    right. left. eexists. split; reflexivity.
  - destruct h; try (right; right; reflexivity).
    left. eexists. eexists. split; reflexivity.
Qed.

(* The value is the one the primitive has. A store has none here, and
   a length has the same at every kind. *)
Lemma fold_named_sound (statics : static_env) (vars : env) (d : named) :
  eval_named statics vars (fold_named d) = eval_named statics vars d.
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try reflexivity.
  destruct (fold_named_prim_cases p args)
    as [(kind & set_kind & -> & ->) | [(kind & -> & ->) | ->]].
  - simpl. destruct (eval_simples vars args); reflexivity.
  - simpl. destruct (eval_simples vars args) as [[ | v [ | w vs]] | ];
      reflexivity.
  - simpl. destruct (prim_may_fold p); [ | reflexivity ].
    destruct (consts_of_simples args) as [cs | ] eqn:Hconsts;
      [ | reflexivity ].
    destruct (eval_prim empty_static_env p (map Value_const cs)) as [v | ]
      eqn:Hprim;
      [ destruct v | ]; try reflexivity.
    simpl. rewrite (consts_of_simples_eval vars args cs Hconsts). simpl.
    rewrite (eval_prim_consts statics empty_static_env), Hprim. reflexivity.
Qed.

(* And it does the same on a heap: a primitive that is folded is not of
   the heap, and its arguments are constants. *)
Lemma fold_named_exec (statics : static_env) (hp : heap) (rd : read_record)
    (vars : env)
    (d : named) :
  exec_named statics hp rd vars (fold_named d)
    = exec_named statics hp rd vars d.
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try reflexivity.
  destruct (fold_named_prim_cases p args)
    as [(kind & set_kind & -> & ->) | [(kind & -> & ->) | ->]].
  - cbn [exec_named].
    destruct (eval_simples vars args) as [vs | ]; [ | reflexivity ].
    destruct
      (FlambdaHeapProofs.array_kind_of_access statics hp vs
         (array_kind_of_set_kind set_kind) kind (Array_kind kind)
         (Array_kind kind) ltac:(constructor) ltac:(constructor) set_kind)
      as (_ & _ & Hset).
    cbv beta iota. rewrite Hset. reflexivity.
  - cbn [exec_named].
    destruct (eval_simples vars args) as [vs | ]; [ | reflexivity ].
    destruct
      (FlambdaHeapProofs.array_kind_of_access statics hp vs Array_values
         Array_values (Array_kind Array_values) kind ltac:(constructor)
         ltac:(constructor) ltac:(constructor))
      as (Hlength & _).
    cbv beta iota. rewrite Hlength. reflexivity.
  - destruct (prim_may_fold p) eqn:Hmay; [ | reflexivity ].
    destruct (consts_of_simples args) as [cs | ] eqn:Hconsts;
      [ | reflexivity ].
    pose proof (consts_of_simples_eval vars args cs Hconsts) as Hargs.
    destruct (eval_prim empty_static_env p (map Value_const cs)) as [v | ]
      eqn:Hprim;
      [ destruct v as [c | | | | | | | | ] | ]; try reflexivity.
    cbv beta iota.
    rewrite (exec_named_not_prim statics hp rd vars (Simple (Const c)))
      by (intros ? ?; discriminate).
    rewrite
      (exec_named_no_location statics hp rd vars p args (map Value_const cs)).
    + simpl. rewrite Hargs. simpl.
      rewrite (eval_prim_consts statics empty_static_env), Hprim.
      reflexivity.
    + intros q ->. discriminate Hmay.
    + exact Hargs.
    + intros l Hl. apply in_map_iff in Hl as (c0 & Hc0 & _).
      discriminate Hc0.
Qed.

Lemma fold_switch_equivalent (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  expr_equivalent (Switch scrutinee arms) (fold_switch scrutinee arms).
Proof.
  unfold fold_switch.
  destruct scrutinee as [x | sym | [i | | | | | | | | | | | | | | ]];
    try (intros vars conts statics hp rd; apply equivalent_refl).
  destruct (find_arm i arms) as [ac | ] eqn:Hfind;
    [ | intros vars conts statics hp rd; apply equivalent_refl ].
  exact (switch_const i arms ac Hfind).
Qed.

Fixpoint fold_constants_equivalent_aux (e : expr) :
  expr_equivalent e (fold_constants e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try (intros vars conts statics hp rd; apply equivalent_refl).
  - apply (expr_equivalent_trans _ (Let x kind (fold_named d) body)).
    + intros vars conts statics hp rd.
      apply let_equivalent_step_exec;
        [ apply fold_named_exec | intros v hp' rd' _ _; apply equivalent_refl ].
    + exact (plug_equivalent (Let_body x kind (fold_named d) Hole) _ _
        (not_many _) (fold_constants_equivalent_aux body)).
  - exact (plug_equivalent (Let_static_body bindings Hole) _ _ (not_many _)
      (fold_constants_equivalent_aux body)).
  - apply (expr_equivalent_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (fold_constants handler)) body)).
    + exact (plug_equivalent
        (Let_cont_non_recursive_handler k params metadata Hole body)
        _ _ (not_many _)
        (fold_constants_equivalent_aux handler)).
    + exact (plug_equivalent
        (Let_cont_non_recursive_body k
           (Handler params metadata (fold_constants handler)) Hole)
        _ _ (not_many _) (fold_constants_equivalent_aux body)).
  - exact (plug_equivalent (Let_cont_recursive_body handlers Hole)
        _ _ (not_many _)
      (fold_constants_equivalent_aux body)).
  - apply fold_switch_equivalent.
Qed.

Theorem fold_constants_equivalent : fold_constants_equivalent_statement.
Proof. exact fold_constants_equivalent_aux. Qed.

(* ======================================================================== *)
(* Removing unused [Let]s whose defining expression always has a value     *)
(* ======================================================================== *)

(* The defining expression has a value of the kind of the [Let], so the
   [Let] takes its step; the body does not look at the variable. *)
Lemma unused_total_let (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  named_is_total kind d = true ->
  ~ var_free_in_expr x body ->
  expr_equivalent (Let x kind d body) body.
Proof.
  intros Htotal Hunused vars conts statics hp rd.
  assert (Hvalue :
    ∃ v : value,
      eval_named statics vars d = Some v
      ∧ kind_eqb (kind_of_value v) kind = true).
  { destruct d as [[y | sym | c] | p args | code slots | fslot decls slots
      | slot closure | pfrom slot closure | move_from move_to closure];
      simpl in Htotal; try discriminate; simpl; eauto. }
  destruct Hvalue as (v & Hd & Hkind).
  assert (Hnot : ∀ (p : prim) (args : list simple), d <> Prim p args).
  { intros p args ->. discriminate Htotal. }
  apply (equivalent_trans _ (State body ((x, v) :: vars) conts statics hp rd));
    [ apply step_equivalent;
      rewrite (step_let_not_prim _ _ _ _ _ _ _ _ rd_ Hnot), Hd, Hkind;
      reflexivity
    | ].
  apply coincide_equivalent.
  - intros y Hy. simpl.
    destruct (Pos.eqb_spec y x) as [-> | _]; [ contradiction | reflexivity ].
  - intros k _. apply (conts_agree_same identity_renaming). reflexivity.
Qed.

Fixpoint remove_unused_total_lets_equivalent_aux (e : expr) :
  expr_equivalent e (remove_unused_total_lets e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try (intros vars conts statics hp rd; apply equivalent_refl).
  - pose proof (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _)
      (remove_unused_total_lets_equivalent_aux body)) as Hbody.
    simpl in Hbody.
    destruct
      (named_is_total kind d
       && negb (var_free_b x (remove_unused_total_lets body)))
      eqn:Hremove; [ | exact Hbody ].
    apply andb_true_iff in Hremove as [Htotal Hunused].
    apply negb_true_iff in Hunused.
    apply (expr_equivalent_trans _ _ _ Hbody), unused_total_let;
      [ exact Htotal | ].
    intros Hx. apply var_free_b_sound in Hx. congruence.
  - exact (plug_equivalent (Let_static_body bindings Hole) _ _ (not_many _)
      (remove_unused_total_lets_equivalent_aux body)).
  - apply (expr_equivalent_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (remove_unused_total_lets handler)) body)).
    + exact (plug_equivalent
        (Let_cont_non_recursive_handler k params metadata Hole body)
        _ _ (not_many _)
        (remove_unused_total_lets_equivalent_aux handler)).
    + exact (plug_equivalent
        (Let_cont_non_recursive_body k
           (Handler params metadata (remove_unused_total_lets handler)) Hole)
        _ _ (not_many _) (remove_unused_total_lets_equivalent_aux body)).
  - exact (plug_equivalent (Let_cont_recursive_body handlers Hole)
        _ _ (not_many _)
      (remove_unused_total_lets_equivalent_aux body)).
Qed.

Theorem remove_unused_total_lets_equivalent :
  remove_unused_total_lets_equivalent_statement.
Proof. exact remove_unused_total_lets_equivalent_aux. Qed.

(* ======================================================================== *)
(* Removing unused [Let]s whose defining expression has a value in scope   *)
(* ======================================================================== *)

(* Each variable in scope is bound to a value of its kind. *)
Definition kinds_hold (kinds : kinds_in_scope) (vars : env) : Prop :=
  ∀ (x : variable) (kind : flambda_kind),
    lookup x kinds = Some kind ->
    ∃ v : value, lookup x vars = Some v ∧ kind_of_value v = kind.

(* Equivalence under the environments that bind the variables given to
   values of their kinds. *)
Definition expr_equivalent_bound (kinds : kinds_in_scope) (e e' : expr)
    : Prop :=
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    kinds_hold kinds vars ->
    equivalent (State e vars conts statics hp rd)
      (State e' vars conts statics hp rd).

Lemma kinds_hold_cons (kinds : kinds_in_scope) (vars : env) (x : variable)
    (v : value) :
  kinds_hold kinds vars ->
  kinds_hold ((x, kind_of_value v) :: kinds) ((x, v) :: vars).
Proof.
  intros Hhold y kind. simpl.
  destruct (Pos.eqb_spec y x) as [-> | Hne];
    [ intros [= <-]; eauto | apply Hhold ].
Qed.

(* The parameters of a handler, bound as a handler that runs binds them,
   hold values of the kinds of the handler. *)
Lemma params_in_scope_hold (params : list variable) :
  ∀ (values : list value) (kinds : kinds_in_scope) (vars handler_vars : env),
    kinds_hold kinds vars ->
    bind_params params values vars = Some handler_vars ->
    kinds_hold (params_in_scope params (map kind_of_value values) kinds)
      handler_vars.
Proof.
  induction params as [ | param params IH];
    intros [ | v values] kinds vars handler_vars Hhold; simpl;
    try discriminate.
  - intros [= <-]. exact Hhold.
  - apply IH, kinds_hold_cons, Hhold.
Qed.

Lemma kind_of_simple_in_scope_sound (kinds : kinds_in_scope) (vars : env)
    (s : simple) (kind : flambda_kind) :
  kinds_hold kinds vars ->
  kind_of_simple_in_scope kinds s = Some kind ->
  ∃ v : value, eval_simple vars s = Some v ∧ kind_of_value v = kind.
Proof.
  intros Hhold. destruct s as [x | sym | c]; simpl.
  - apply Hhold.
  - intros [= <-]. eauto.
  - intros [= <-]. eauto.
Qed.

Lemma simple_in_scope_defined (kinds : kinds_in_scope) (vars : env)
    (s : simple) :
  kinds_hold kinds vars ->
  simple_in_scope (map fst kinds) s = true ->
  ∃ v : value, eval_simple vars s = Some v.
Proof.
  intros Hhold. destruct s as [x | sym | c]; simpl; eauto.
  intros Hin. apply among_In in Hin.
  destruct (lookup x kinds) as [kind | ] eqn:Hlookup;
    [ | destruct (lookup_none_not_in x kinds Hlookup Hin) ].
  destruct (Hhold x kind Hlookup) as (v & Hv & _). eauto.
Qed.

Lemma value_slots_defined (kinds : kinds_in_scope) (vars : env)
    (slots : list (value_slot * simple)) :
  kinds_hold kinds vars ->
  forallb (fun slot => simple_in_scope (map fst kinds) (snd slot)) slots
  = true ->
  ∃ evaluated : list (value_slot * value),
    eval_value_slots vars slots = Some evaluated.
Proof.
  intros Hhold. induction slots as [ | [slot s] slots IH]; simpl; [ eauto | ].
  intros Hscope. apply andb_true_iff in Hscope as [Hs Hslots].
  destruct (simple_in_scope_defined kinds vars s Hhold Hs) as [v ->].
  destruct (IH Hslots) as [evaluated ->]. simpl. eauto.
Qed.

Lemma named_defined_sound (kinds : kinds_in_scope) (vars : env)
    (statics : static_env) (kind : flambda_kind) (d : named) :
  kinds_hold kinds vars ->
  named_defined kinds kind d = true ->
  ∃ v : value,
    eval_named statics vars d = Some v
    ∧ kind_eqb (kind_of_value v) kind = true.
Proof.
  intros Hhold.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    simpl; try discriminate.
  - destruct (kind_of_simple_in_scope kinds s) as [found | ] eqn:Hs;
      [ | discriminate ].
    intros Hkind.
    destruct (kind_of_simple_in_scope_sound kinds vars s found Hhold Hs)
      as (v & Hv & <-).
    eauto.
  - intros Hdefined. apply andb_true_iff in Hdefined as [Hkind Hslots].
    destruct (value_slots_defined kinds vars slots Hhold Hslots)
      as [evaluated ->].
    simpl. eauto.
  - intros Hdefined. apply andb_true_iff in Hdefined as [Hdefined Hslots].
    apply andb_true_iff in Hdefined as [Hkind Hslot].
    apply is_bound_lookup in Hslot as [code ->].
    destruct (value_slots_defined kinds vars slots Hhold Hslots)
      as [evaluated ->].
    simpl. eauto.
Qed.

Lemma let_equivalent_bound (kinds : kinds_in_scope) (x : variable)
    (kind : flambda_kind) (d : named) (body body' : expr) :
  expr_equivalent_bound ((x, kind) :: kinds) body body' ->
  expr_equivalent_bound kinds (Let x kind d body) (Let x kind d body').
Proof.
  intros Hbody vars conts statics hp rd Hhold.
  apply let_equivalent_step_exec; [ reflexivity | intros v hp' rd' _ <- ].
  apply Hbody, kinds_hold_cons, Hhold.
Qed.

(* The defining expression has a value of the kind of the [Let], so the
   [Let] takes its step; the body does not look at the variable. *)
Lemma unused_defined_let (kinds : kinds_in_scope) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  named_defined kinds kind d = true ->
  ~ var_free_in_expr x body ->
  expr_equivalent_bound kinds (Let x kind d body) body.
Proof.
  intros Hdefined Hunused vars conts statics hp rd Hhold.
  destruct (named_defined_sound kinds vars statics kind d Hhold Hdefined)
    as (v & Hd & Hkind).
  assert (Hnot : ∀ (p : prim) (args : list simple), d <> Prim p args).
  { intros p args ->. discriminate Hdefined. }
  apply (equivalent_trans _ (State body ((x, v) :: vars) conts statics hp rd));
    [ apply step_equivalent;
      rewrite (step_let_not_prim _ _ _ _ _ _ _ _ rd_ Hnot), Hd, Hkind;
      reflexivity
    | ].
  apply coincide_equivalent.
  - intros y Hy. simpl.
    destruct (Pos.eqb_spec y x) as [-> | _]; [ contradiction | reflexivity ].
  - intros k _. apply (conts_agree_same identity_renaming). reflexivity.
Qed.

Lemma let_cont_handler_equivalent_bound (kinds : kinds_in_scope)
    (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' body : expr) :
  expr_equivalent_bound
    (params_in_scope params (hm_params_kinds metadata) kinds) handler
    handler' ->
  expr_equivalent_bound kinds
    (Let_cont_non_recursive k (Handler params metadata handler) body)
    (Let_cont_non_recursive k (Handler params metadata handler') body).
Proof.
  intros Hhandler vars conts statics hp rd Hhold.
  assert (Hruns :
    ∀ (values : list value) (handler_vars : env),
      map kind_of_value values = hm_params_kinds metadata ->
      bind_params params values vars = Some handler_vars ->
      kinds_hold (params_in_scope params (hm_params_kinds metadata) kinds)
        handler_vars).
  { intros values handler_vars <-.
    exact (params_in_scope_hold params values kinds vars handler_vars Hhold). }
  apply refines_antisymmetric; apply let_cont_handler_refines_at;
    intros values handler_vars conts' statics' hp' rd' Hkinds Hbind;
    apply equivalent_refines; [ | apply equivalent_sym ];
    apply Hhandler, (Hruns values handler_vars Hkinds Hbind).
Qed.

Lemma expr_equivalent_bound_trans (kinds : kinds_in_scope)
    (e e' e'' : expr) :
  expr_equivalent_bound kinds e e' -> expr_equivalent_bound kinds e' e'' ->
  expr_equivalent_bound kinds e e''.
Proof.
  intros H H' vars conts statics hp rd Hhold.
  exact (equivalent_trans _ _ _ (H vars conts statics hp rd_ Hhold)
    (H' vars conts statics hp rd_ Hhold)).
Qed.

Fixpoint remove_unused_defined_lets_equivalent_bound (e : expr) :
  ∀ kinds : kinds_in_scope,
    expr_equivalent_bound kinds e (remove_unused_defined_lets kinds e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros kinds; simpl;
    try (intros vars conts statics hp rd _; apply equivalent_refl).
  - pose proof (let_equivalent_bound kinds x kind d _ _
      (remove_unused_defined_lets_equivalent_bound body
         ((x, kind) :: kinds)))
      as Hbody.
    destruct
      (named_defined kinds kind d
       && negb
            (var_free_b x
               (remove_unused_defined_lets ((x, kind) :: kinds) body)))
      eqn:Hremove; [ | exact Hbody ].
    apply andb_true_iff in Hremove as [Hdefined Hunused].
    apply negb_true_iff in Hunused.
    apply (expr_equivalent_bound_trans kinds _ _ _ Hbody),
      unused_defined_let;
      [ exact Hdefined | ].
    intros Hx. apply var_free_b_sound in Hx. congruence.
  - intros vars conts statics hp rd Hhold.
    apply let_static_equivalent_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ];
      [ | exact I ].
    apply (remove_unused_defined_lets_equivalent_bound body kinds), Hhold.
  - apply (expr_equivalent_bound_trans kinds _
      (Let_cont_non_recursive k
         (Handler params metadata
            (remove_unused_defined_lets
               (params_in_scope params (hm_params_kinds metadata) kinds)
               handler))
         body)).
    + apply let_cont_handler_equivalent_bound,
        remove_unused_defined_lets_equivalent_bound.
    + intros vars conts statics hp rd Hhold.
      apply let_cont_non_recursive_equivalent_step.
      apply (remove_unused_defined_lets_equivalent_bound body kinds), Hhold.
  - intros vars conts statics hp rd Hhold.
    apply let_cont_recursive_equivalent_step; [ reflexivity | intros _ ].
    apply (remove_unused_defined_lets_equivalent_bound body kinds), Hhold.
Qed.

Theorem remove_unused_defined_lets_equivalent :
  remove_unused_defined_lets_equivalent_statement.
Proof.
  intros e vars conts statics hp rd.
  apply (remove_unused_defined_lets_equivalent_bound e []).
  intros x kind [=].
Qed.

(* ======================================================================== *)
(* Inlining calls of symbols known to be closures                           *)
(* ======================================================================== *)

(* The static environment binds the symbols known to be closures to
   closures of the code known. *)
Definition statics_know_symbols (known : list (symbol * option code_id))
    (statics : static_env) : Prop :=
  ∀ (s : symbol) (code : code_id),
    lookup s known = Some (Some code) ->
    ∃ slots : list (value_slot * value),
      closure_of statics (Value_symbol s) = Some (code, slots).

Lemma eval_static_bindings_know_symbols (vars : env)
    (bindings : list static_binding) :
  ∀ (known : list (symbol * option code_id)) (statics extended : static_env),
    statics_know_symbols known statics ->
    eval_static_bindings vars bindings statics = Some extended ->
    statics_know_symbols (symbols_of bindings known) extended.
Proof.
  induction bindings as [ | binding bindings IH];
    intros known statics extended Hknow; simpl.
  - intros [= <-]. exact Hknow.
  - destruct binding
      as [id c | id
         | sym
             [code slots | fslot syms decls slots | tag fields | contents
             | bytes | elements]];
      simpl.
    + destruct (lookup id (static_code statics)); [ discriminate | ].
      apply IH. exact Hknow.
    + apply IH, Hknow.
    + destruct (lookup sym (static_values statics)); [ discriminate | ].
      destruct (eval_value_slots vars slots) as [evaluated | ]; simpl;
        [ | discriminate ].
      apply IH. intros s code'. simpl.
      destruct (Pos.eqb s sym); [ intros [= <-]; eauto | apply Hknow ].
    + destruct (lookup sym (static_values statics)); [ discriminate | ].
      destruct (lookup fslot decls) as [code | ] eqn:Hcode; simpl;
        [ | discriminate ].
      destruct (eval_value_slots vars slots) as [evaluated | ]; simpl;
        [ | discriminate ].
      apply IH. intros s code'. simpl.
      destruct (Pos.eqb s sym); [ | apply Hknow ].
      intros [= <-]. simpl. rewrite Hcode. simpl. eauto.
    + destruct (lookup sym (static_values statics)); [ discriminate | ].
      destruct (eval_simples vars fields) as [vs | ]; simpl;
        [ | discriminate ].
      apply IH. intros s code'. simpl.
      destruct (Pos.eqb s sym); [ discriminate | apply Hknow ].
    + destruct (lookup sym (static_values statics)); [ discriminate | ].
      destruct (eval_simple vars contents) as [v | ]; simpl;
        [ | discriminate ].
      apply IH. intros s code'. simpl.
      destruct (Pos.eqb s sym); [ discriminate | apply Hknow ].
    + destruct (lookup sym (static_values statics)); [ discriminate | ].
      apply IH. intros s code'. simpl.
      destruct (Pos.eqb s sym); [ discriminate | apply Hknow ].
    + destruct (lookup sym (static_values statics)); [ discriminate | ].
      destruct (eval_simples vars elements) as [vs | ]; simpl;
        [ | discriminate ].
      apply IH. intros s code'. simpl.
      destruct (Pos.eqb s sym); [ discriminate | apply Hknow ].
Qed.

Section Kept.

(* The code whose calls are left alone. *)
Variable kept : list code_id.

Lemma inline_known_apply_equivalent (known_code : list (code_id * code))
    (known_symbols : list (symbol * option code_id)) (ap : apply_expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  statics_know known_code statics ->
  statics_know_symbols known_symbols statics ->
  equivalent (State (Apply ap) vars conts statics hp rd)
    (State (inline_known_apply kept known_code known_symbols ap) vars conts
       statics hp rd).
Proof.
  intros Hcode Hsymbols. unfold inline_known_apply.
  destruct (ap_callee ap) as [x | s | const] eqn:Hcallee;
    try apply equivalent_refl.
  destruct (ap_call_kind ap) as [called | | code_ids] eqn:Hkind;
    [ | apply equivalent_refl | apply equivalent_refl ].
  destruct (lookup s known_symbols) as [[code | ] | ] eqn:Hs;
    try apply equivalent_refl.
  destruct (lookup called known_code) as [c | ] eqn:Hc;
    [ | apply equivalent_refl ].
  destruct
    (negb (among called kept)
     && (Pos.eqb code called && inlinable_b ap c && call_agrees_b ap c))
    eqn:Hcheck;
    [ | apply equivalent_refl ].
  apply andb_true_iff in Hcheck as [_ Hcheck].
  apply andb_true_iff in Hcheck as [Hcheck Hagrees].
  apply andb_true_iff in Hcheck as [Hsame Hinlinable].
  apply Pos.eqb_eq in Hsame. subst code.
  unfold call_agrees_b in Hagrees.
  apply andb_true_iff in Hagrees as [Hagrees Hparams_kinds].
  apply andb_true_iff in Hagrees as [Hargs_arity Hreturn_arity].
  apply kinds_per_parameter_eqb_eq in Hargs_arity.
  apply arity_eqb_eq in Hreturn_arity.
  apply Nat.eqb_eq in Hparams_kinds.
  destruct (Hsymbols s called Hs) as [slots Hclosure].
  apply (inline_equivalent ap called c _ (Value_symbol s) slots);
    [ exact Hkind | rewrite Hcallee; reflexivity
    | exact Hclosure | exact (Hcode called c Hc)
    | exact (inlinable_b_sound ap c Hinlinable) | exact Hargs_arity
    | exact Hreturn_arity | exact Hparams_kinds ].
Qed.

Definition statics_know_both (known_code : list (code_id * code))
    (known_symbols : list (symbol * option code_id)) (statics : static_env)
    : Prop :=
  statics_know known_code statics
  ∧ statics_know_symbols known_symbols statics.

Lemma statics_know_both_extend (known_code : list (code_id * code))
    (known_symbols : list (symbol * option code_id))
    (statics statics' : static_env) :
  statics_know_both known_code known_symbols statics ->
  statics_extend statics statics' ->
  statics_know_both known_code known_symbols statics'.
Proof.
  intros [Hcode Hsymbols] Hextend. split.
  - exact (statics_know_extend known_code statics statics' Hcode Hextend).
  - intros s code Hs. destruct (Hsymbols s code Hs) as [slots Hclosure].
    exists slots. revert Hclosure. simpl.
    destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hlookup;
      [ | discriminate ].
    rewrite (proj2 Hextend s allocated Hlookup). auto.
Qed.

Lemma inline_known_apply_binds_statics
    (known_code : list (code_id * code))
    (known_symbols : list (symbol * option code_id)) (ap : apply_expr) :
  binds_statics (inline_known_apply kept known_code known_symbols ap) = false.
Proof.
  unfold inline_known_apply.
  destruct (ap_callee ap) as [x | s | const]; try reflexivity.
  destruct (ap_call_kind ap) as [called | | code_ids]; try reflexivity.
  destruct (lookup s known_symbols) as [[code | ] | ]; try reflexivity.
  destruct (lookup called known_code) as [c | ]; try reflexivity.
  destruct
    (negb (among called kept)
     && (Pos.eqb code called && inlinable_b ap c && call_agrees_b ap c))
    eqn:Hcheck;
    [ | reflexivity ].
  apply andb_true_iff in Hcheck as [_ Hcheck].
  apply andb_true_iff in Hcheck as [Hcheck _].
  apply andb_true_iff in Hcheck as [_ Hinlinable].
  rewrite inlined_body_binds_statics.
  exact (inlinable_b_binds_statics ap c Hinlinable).
Qed.

Fixpoint inline_known_calls_binds_statics (e : expr) :
  ∀ (known_code : list (code_id * code))
    (known_symbols : list (symbol * option code_id)),
    binds_statics (inline_known_calls kept known_code known_symbols e)
    = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros known_code known_symbols; simpl; try reflexivity.
  - apply inline_known_calls_binds_statics.
  - rewrite inline_known_calls_binds_statics. f_equal.
    induction bindings as [ | [id c | id | sym const] bindings IH]; simpl;
      congruence.
  - destruct h as [params metadata handler]. simpl.
    rewrite !inline_known_calls_binds_statics. reflexivity.
  - rewrite inline_known_calls_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite inline_known_calls_binds_statics, IH. reflexivity.
  - apply inline_known_apply_binds_statics.
Qed.

Lemma inline_known_calls_handlers_bind_statics
    (known_code : list (code_id * code))
    (known_symbols : list (symbol * option code_id))
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map
       (fun '(k, h) =>
          (k, inline_known_calls_in_handler kept known_code known_symbols h))
       handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
    [ reflexivity | ].
  rewrite inline_known_calls_binds_statics, IH. reflexivity.
Qed.

Fixpoint inline_known_calls_equivalent_knowing (e : expr) :
  ∀ (known_code : list (code_id * code))
    (known_symbols : list (symbol * option code_id)) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record),
    statics_know known_code statics ->
    statics_know_symbols known_symbols statics ->
    equivalent (State e vars conts statics hp rd)
      (State (inline_known_calls kept known_code known_symbols e) vars conts
         statics hp rd).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body
         | ap | ac | scrutinee arms | ];
      intros known_code known_symbols vars conts statics hp rd Hcode Hsymbols;
      simpl; try apply equivalent_refl.
    + apply let_equivalent_step_exec; [ reflexivity | intros v hp' rd' _ _ ].
      apply inline_known_calls_equivalent_knowing; assumption.
    + assert (Hrefine :
        let bindings' :=
          map (inline_known_calls_in_binding kept known_code known_symbols)
            bindings
        in
        bindings_refine (statics_know_both known_code known_symbols)
          bindings bindings'
        ∧ bindings_refine (statics_know_both known_code known_symbols)
            bindings' bindings).
      { induction bindings
          as [ | [id [params my_closure ret exn metadata code_body] | id
                 | sym const]
                 bindings IH];
          simpl.
        - repeat split; apply Forall2_nil.
        - destruct IH as [[Hf Hb] [Hf' Hb']].
          repeat split; (apply Forall2_cons; [ | assumption ]);
            apply Binding_code, Codes_related;
            first
              [ apply inline_known_calls_binds_statics
              | symmetry; apply inline_known_calls_binds_statics
              | intros vars' conts' statics' hp' rd' [Hcode' Hsymbols'];
                first
                  [ apply equivalent_refines,
                      inline_known_calls_equivalent_knowing; assumption
                  | apply equivalent_refines, equivalent_sym,
                      inline_known_calls_equivalent_knowing; assumption ] ].
        - destruct IH as [[Hf Hb] [Hf' Hb']].
          repeat split;
            (apply Forall2_cons; [ apply Binding_same | assumption ]).
        - destruct IH as [[Hf Hb] [Hf' Hb']].
          repeat split;
            (apply Forall2_cons; [ apply Binding_same | assumption ]). }
      remember
        (map (inline_known_calls_in_binding kept known_code known_symbols)
           bindings)
        as bindings' eqn:Hdef.
      clear Hdef. simpl in Hrefine.
      destruct Hrefine as [[Hf Hb] [Hf' Hb']].
      apply (fun H =>
        eval_static_bindings_related _ vars _ _ H statics statics
          (statics_related_refl _ statics)) in Hf, Hb, Hf', Hb'.
      apply let_static_equivalent_step.
      destruct (eval_static_bindings vars bindings statics) as [extended | ]
        eqn:Hbindings,
        (eval_static_bindings vars bindings' statics) as [extended' | ]
          eqn:Hbindings';
        try contradiction; [ | exact I ].
      assert (Hok : statics_know_both known_code known_symbols statics)
        by (split; assumption).
      pose proof (statics_know_both_extend known_code known_symbols _ _
        Hok (eval_static_bindings_extend vars _ _ _ Hbindings)) as Hext.
      pose proof (statics_know_both_extend known_code known_symbols _ _
        Hok (eval_static_bindings_extend vars _ _ _ Hbindings')) as Hext'.
      apply (equivalent_trans _ (State body vars conts extended' hp rd)).
      * apply refines_antisymmetric;
          apply (related_states_refines
            (statics_know_both known_code known_symbols)
            (statics_know_both_extend known_code known_symbols));
          try assumption; apply cont_env_related_refl.
      * apply inline_known_calls_equivalent_knowing.
        -- exact (eval_static_bindings_know vars bindings' known_code
             statics extended' Hcode Hbindings').
        -- exact (eval_static_bindings_know_symbols vars bindings'
             known_symbols statics extended' Hsymbols Hbindings').
    + destruct h as [params metadata handler]. simpl.
      apply (equivalent_trans _
        (State
           (Let_cont_non_recursive k
              (Handler params metadata
                 (inline_known_calls kept known_code known_symbols handler))
              body)
           vars conts statics hp rd)).
      * apply refines_antisymmetric;
          apply (let_cont_handler_refines_where
            (statics_know_both known_code known_symbols)
            (statics_know_both_extend known_code known_symbols));
          try (split; assumption);
          intros vs handler_vars conts' statics' hp' rd' [Hcode' Hsymbols'] _ _;
          apply equivalent_refines; [ | apply equivalent_sym ];
          apply inline_known_calls_equivalent_knowing; assumption.
      * apply let_cont_non_recursive_equivalent_step.
        apply inline_known_calls_equivalent_knowing; assumption.
    + apply (equivalent_trans _
        (State
           (Let_cont_recursive
              (map
                 (fun '(k, h) =>
                    (k,
                     inline_known_calls_in_handler kept known_code known_symbols
                       h))
                 handlers)
              body)
           vars conts statics hp rd)).
      * assert (Hhandlers :
          let handlers' :=
            map
              (fun '(k, h) =>
                 (k,
                  inline_known_calls_in_handler kept known_code known_symbols
                    h))
              handlers
          in
          handlers_refine (statics_know_both known_code known_symbols) vars
            handlers handlers'
          ∧ handlers_refine (statics_know_both known_code known_symbols)
              vars handlers' handlers).
        { induction handlers
            as [ | [k [params metadata handler]] handlers IH];
            simpl.
          - repeat split; apply Forall2_nil.
          - destruct IH as [[Hforwards Hbackwards] [Hforwards' Hbackwards']].
            repeat split; (apply Forall2_cons; [ | assumption ]);
              (split; [ reflexivity | ]); apply Handler_related;
              intros vs handler_vars conts' statics' hp' rd' [Hcode' Hsymbols']
                _ _;
              try (apply equivalent_refines,
                     inline_known_calls_equivalent_knowing; assumption);
              apply equivalent_refines, equivalent_sym,
                inline_known_calls_equivalent_knowing; assumption. }
        destruct Hhandlers
          as [[Hforwards Hbackwards] [Hforwards' Hbackwards']].
        apply refines_antisymmetric;
          apply (let_cont_recursive_handlers_refine_where
            (statics_know_both known_code known_symbols)
            (statics_know_both_extend known_code known_symbols));
          try assumption; try (split; assumption);
          first
            [ apply inline_known_calls_handlers_bind_statics
            | symmetry; apply inline_known_calls_handlers_bind_statics ].
      * apply let_cont_recursive_equivalent_step; [ reflexivity | intros _ ].
        apply inline_known_calls_equivalent_knowing; assumption.
    + apply inline_known_apply_equivalent; assumption.
Qed.

End Kept.

Theorem inline_known_calls_equivalent :
  inline_known_calls_equivalent_statement.
Proof.
  intros kept e vars conts statics hp rd.
  apply inline_known_calls_equivalent_knowing.
  - intros id c Hlookup. discriminate Hlookup.
  - intros s code Hlookup. discriminate Hlookup.
Qed.

(* ======================================================================== *)
(* Groups of static bindings                                                *)
(* ======================================================================== *)

(* A binding of a symbol may be made after bindings of code that came
   after it: code and symbols go to different parts of the static
   environment. *)
Lemma eval_static_bindings_symbol_later (vars : env) (sym : symbol)
    (const : static_const) (codes rest : list static_binding) :
  forallb binds_code codes = true ->
  ∀ statics : static_env,
    eval_static_bindings vars (codes ++ Bind_symbol sym const :: rest) statics
    = eval_static_bindings vars (Bind_symbol sym const :: codes ++ rest)
        statics.
Proof.
  induction codes as [ | binding codes IH]; intros Hcodes statics;
    [ reflexivity | ].
  simpl in Hcodes. apply andb_true_iff in Hcodes as [Hbinding Hcodes].
  destruct binding as [id c | id | sym' const']; simpl in Hbinding;
    try discriminate.
  - change
      (match lookup id (static_code statics) with
       | Some _ => None
       | None =>
         eval_static_bindings vars (codes ++ Bind_symbol sym const :: rest)
           (Static_env ((id, c) :: static_code statics)
              (static_values statics))
       end
       = eval_static_bindings vars
           (Bind_symbol sym const :: Bind_code id c :: codes ++ rest)
           statics).
    rewrite (IH Hcodes). simpl.
    destruct (lookup id (static_code statics)),
      (lookup sym (static_values statics)),
      (eval_static_const vars const); reflexivity.
  - change
      (eval_static_bindings vars (codes ++ Bind_symbol sym const :: rest)
         statics
       = eval_static_bindings vars
           (Bind_symbol sym const :: Bind_deleted_code id :: codes ++ rest)
           statics).
    rewrite (IH Hcodes). simpl.
    destruct (lookup sym (static_values statics)),
      (eval_static_const vars const); reflexivity.
Qed.

Lemma filter_binds_code (bindings : list static_binding) :
  forallb binds_code (filter binds_code bindings) = true.
Proof.
  induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
  destruct (binds_code binding) eqn:Hcode; simpl; [ rewrite Hcode | ];
    exact IH.
Qed.

Lemma eval_static_bindings_code_first (vars : env)
    (bindings : list static_binding) :
  ∀ statics : static_env,
    eval_static_bindings vars (code_first bindings) statics
    = eval_static_bindings vars bindings statics.
Proof.
  unfold code_first.
  induction bindings as [ | binding bindings IH]; intros statics;
    [ reflexivity | ].
  destruct binding as [id c | id | sym const]; simpl.
  - destruct (lookup id (static_code statics)); [ reflexivity | apply IH ].
  - apply IH.
  - rewrite (eval_static_bindings_symbol_later vars sym const _ _
      (filter_binds_code bindings)).
    simpl. destruct (lookup sym (static_values statics)); [ reflexivity | ].
    destruct (eval_static_const vars const); simpl;
      [ apply IH | reflexivity ].
Qed.

Lemma code_first_equivalent (bindings : list static_binding) (body : expr) :
  expr_equivalent (Let_static bindings body)
    (Let_static (code_first bindings) body).
Proof.
  intros vars conts statics hp rd. apply step_eq_equivalent.
  rewrite !step_let_static, eval_static_bindings_code_first. reflexivity.
Qed.

Fixpoint merge_statics_equivalent_aux (e : expr) :
  expr_equivalent e (merge_statics e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try (intros vars conts statics hp rd; apply equivalent_refl).
  - exact (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _)
      (merge_statics_equivalent_aux body)).
  - pose proof (plug_equivalent (Let_static_body bindings Hole) _ _ (not_many _)
      (merge_statics_equivalent_aux body)) as Hbody.
    simpl in Hbody.
    destruct (merge_statics body)
      as [ x kind' d body' | bindings' body' | k h body' | handlers body' | ap
         | ac | scrutinee arms | ];
      try (apply (expr_equivalent_trans _ _ _ Hbody), code_first_equivalent).
    apply (expr_equivalent_trans _ _ _ Hbody).
    apply (expr_equivalent_trans _
      (Let_static (bindings ++ bindings') body')).
    + intros vars conts statics hp rd.
      apply equivalent_sym, let_static_split.
    + apply code_first_equivalent.
  - apply (expr_equivalent_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (merge_statics handler)) body)).
    + exact (plug_equivalent
        (Let_cont_non_recursive_handler k params metadata Hole body)
        _ _ (not_many _)
        (merge_statics_equivalent_aux handler)).
    + exact (plug_equivalent
        (Let_cont_non_recursive_body k
           (Handler params metadata (merge_statics handler)) Hole)
        _ _ (not_many _) (merge_statics_equivalent_aux body)).
  - exact (plug_equivalent (Let_cont_recursive_body handlers Hole)
        _ _ (not_many _)
      (merge_statics_equivalent_aux body)).
Qed.

Theorem merge_statics_equivalent : merge_statics_equivalent_statement.
Proof. exact merge_statics_equivalent_aux. Qed.

(* ======================================================================== *)
(* Switches whose arms are all the same                                     *)
(* ======================================================================== *)

Lemma collapse_switch_refines (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  expr_refines (Switch scrutinee arms) (collapse_switch scrutinee arms).
Proof.
  unfold collapse_switch.
  destruct arms as [ | [i ac] others]; [ apply expr_refines_refl | ].
  destruct
    (forallb
       (fun arm => if apply_cont_eq_dec (snd arm) ac then true else false)
       others)
    eqn:Hsame; [ | apply expr_refines_refl ].
  apply switch_same_arms. intros j arm [[= _ <-] | Hin]; [ reflexivity | ].
  rewrite forallb_forall in Hsame. specialize (Hsame (j, arm) Hin).
  simpl in Hsame.
  destruct (apply_cont_eq_dec arm ac); [ assumption | discriminate ].
Qed.

Fixpoint collapse_switches_refines_aux (e : expr) :
  expr_refines e (collapse_switches e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try apply expr_refines_refl.
  - exact (plug_refines (Let_body x kind d Hole) _ _ (not_many _)
      (collapse_switches_refines_aux body)).
  - exact (plug_refines (Let_static_body bindings Hole) _ _ (not_many _)
      (collapse_switches_refines_aux body)).
  - apply (expr_refines_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (collapse_switches handler)) body)).
    + exact (plug_refines
        (Let_cont_non_recursive_handler k params metadata Hole body)
        _ _ (not_many _)
        (collapse_switches_refines_aux handler)).
    + exact (plug_refines
        (Let_cont_non_recursive_body k
           (Handler params metadata (collapse_switches handler)) Hole)
        _ _ (not_many _) (collapse_switches_refines_aux body)).
  - exact (plug_refines (Let_cont_recursive_body handlers Hole) _ _ (not_many _)
      (collapse_switches_refines_aux body)).
  - apply collapse_switch_refines.
Qed.

Theorem collapse_switches_refines : collapse_switches_refines_statement.
Proof. exact collapse_switches_refines_aux. Qed.

(* ======================================================================== *)
(* Rewriting the bodies of code                                             *)
(* ======================================================================== *)

Fixpoint rewrite_code_binds_statics (f : expr -> expr) (e : expr) :
  binds_statics (rewrite_code f e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - apply rewrite_code_binds_statics.
  - rewrite rewrite_code_binds_statics. f_equal.
    induction bindings
      as [ | [id [params my_closure ret exn metadata code_body] | id
             | sym const]
             bindings IH];
      simpl; congruence.
  - destruct h as [params metadata handler]. simpl.
    rewrite !rewrite_code_binds_statics. reflexivity.
  - rewrite rewrite_code_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite rewrite_code_binds_statics, IH. reflexivity.
Qed.

Lemma rewrite_code_handlers_bind_statics (f : expr -> expr)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map (fun '(k, h) => (k, rewrite_code_in_handler f h)) handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
    [ reflexivity | ].
  rewrite rewrite_code_binds_statics, IH. reflexivity.
Qed.

(* A group of bindings may be replaced by one whose code has bodies that
   refine. *)
Lemma let_static_bindings_refine (bindings bindings' : list static_binding)
    (body : expr) :
  bindings_refine always bindings bindings' ->
  expr_refines (Let_static bindings body) (Let_static bindings' body).
Proof.
  intros [Hforwards Hbackwards] vars conts statics hp rd.
  apply (fun H =>
    eval_static_bindings_related _ vars _ _ H statics statics
      (statics_related_refl _ statics)) in Hforwards, Hbackwards.
  apply let_static_refines_step.
  destruct (eval_static_bindings vars bindings statics) as [extended | ],
    (eval_static_bindings vars bindings' statics) as [extended' | ];
    try contradiction; [ | exact I ].
  apply (related_states_refines always always_extend);
    [ exact I | exact I | apply cont_env_related_refl
    | apply cont_env_related_refl | exact Hforwards | exact Hbackwards ].
Qed.

Lemma rewrite_bindings_refine (f : expr -> expr)
    (bindings : list static_binding) :
  (∀ e : expr, expr_refines e (f e)) ->
  bindings_refine always bindings (map (rewrite_code_in_binding f) bindings).
Proof.
  intros Hf.
  induction bindings
    as [ | [id [params my_closure ret exn metadata code_body] | id
           | sym const]
           bindings [Hforwards Hbackwards]];
    simpl.
  - split; apply Forall2_nil.
  - unfold rewrite_code_body.
    destruct (Bool.eqb (binds_statics code_body)
                (binds_statics (f code_body))) eqn:Hbinds;
      [ apply eqb_prop in Hbinds | ];
      split; (apply Forall2_cons; [ | assumption ]);
      try apply Binding_same;
      apply Binding_code, Codes_related;
      first
        [ exact Hbinds | exact (eq_sym Hbinds)
        | intros vars conts statics hp rd _; apply Hf ].
  - split; (apply Forall2_cons; [ apply Binding_same | assumption ]).
  - split; (apply Forall2_cons; [ apply Binding_same | assumption ]).
Qed.

Lemma rewrite_bindings_refined (f : expr -> expr)
    (bindings : list static_binding) :
  (∀ e : expr, expr_refines (f e) e) ->
  bindings_refine always (map (rewrite_code_in_binding f) bindings) bindings.
Proof.
  intros Hf.
  induction bindings
    as [ | [id [params my_closure ret exn metadata code_body] | id
           | sym const]
           bindings [Hforwards Hbackwards]];
    simpl.
  - split; apply Forall2_nil.
  - unfold rewrite_code_body.
    destruct (Bool.eqb (binds_statics code_body)
                (binds_statics (f code_body))) eqn:Hbinds;
      [ apply eqb_prop in Hbinds | ];
      split; (apply Forall2_cons; [ | assumption ]);
      try apply Binding_same;
      apply Binding_code, Codes_related;
      first
        [ exact Hbinds | exact (eq_sym Hbinds)
        | intros vars conts statics hp rd _; apply Hf ].
  - split; (apply Forall2_cons; [ apply Binding_same | assumption ]).
  - split; (apply Forall2_cons; [ apply Binding_same | assumption ]).
Qed.

(* The same script proves the rewrite in the two directions: [IH] is the
   induction, and [Hf] what is assumed of the rewrite. *)
Ltac rewrite_code_cases IH Hf :=
  match goal with
  | |- expr_refines (Let ?x ?kind ?d ?body) (Let _ _ _ ?body') =>
    first
      [ exact
          (plug_refines (Let_body x kind d Hole) _ _ (not_many _) (IH body))
      | exact
          (plug_refines (Let_body x kind d Hole) _ _ (not_many _) (IH body'))
      ]
  | |- expr_refines (Let_static _ ?body) (Let_static _ ?body') =>
    eapply expr_refines_trans;
      [ apply let_static_bindings_refine;
        first
          [ apply rewrite_bindings_refine, Hf
          | apply rewrite_bindings_refined, Hf ]
      | first
          [ exact (plug_refines (Let_static_body _ Hole) _ _ (not_many _)
              (IH body))
          | exact (plug_refines (Let_static_body _ Hole) _ _ (not_many _)
              (IH body')) ] ]
  | |- expr_refines
         (Let_cont_non_recursive ?k (Handler ?params ?metadata ?handler) _)
         (Let_cont_non_recursive _ (Handler _ _ ?handler') _) =>
    eapply expr_refines_trans;
      [ intros vars conts statics hp rd; apply let_cont_handler_refines_at;
        intros vs handler_vars conts' statics' hp' rd' _ _;
        first [ apply (IH handler) | apply (IH handler') ]
      | match goal with
        | |- expr_refines (Let_cont_non_recursive _ ?h ?body) _ =>
          first
            [ exact (plug_refines (Let_cont_non_recursive_body k h Hole) _ _
                (not_many _) (IH body))
            | match goal with
              | |- expr_refines _ (Let_cont_non_recursive _ _ ?body') =>
                exact (plug_refines (Let_cont_non_recursive_body k h Hole)
                  _ _ (not_many _) (IH body'))
              end ]
        end ]
  end.

Fixpoint rewrite_code_refines_aux (f : expr -> expr)
    (Hf : ∀ e : expr, expr_refines e (f e)) (e : expr) {struct e} :
  expr_refines e (rewrite_code f e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    simpl; try apply expr_refines_refl;
    try (destruct h as [params metadata handler]; simpl);
    try (rewrite_code_cases (rewrite_code_refines_aux f Hf) Hf; fail).
  assert (Hhandlers :
    ∀ vars : env,
      handlers_refine always vars handlers
        (map (fun '(k, h) => (k, rewrite_code_in_handler f h)) handlers)).
  { induction handlers as [ | [k [params metadata handler]] handlers IH];
      intros vars; simpl.
    - split; apply Forall2_nil.
    - destruct (IH vars) as [Hforwards Hbackwards].
      split; (apply Forall2_cons; [ | assumption ]);
        (split; [ reflexivity | ]); apply Handler_related;
        intros vs handler_vars conts statics hp rd _ _ _;
        apply (rewrite_code_refines_aux f Hf handler). }
  eapply expr_refines_trans;
    [ | exact (plug_refines (Let_cont_recursive_body _ Hole) _ _ (not_many _)
          (rewrite_code_refines_aux f Hf body)) ].
  intros vars conts statics hp rd.
  destruct (Hhandlers vars) as [Hforwards Hbackwards].
  apply (let_cont_recursive_handlers_refine_where always always_extend);
    [ exact I | symmetry; apply rewrite_code_handlers_bind_statics
    | exact Hforwards | exact Hbackwards ].
Qed.

Fixpoint rewrite_code_refined_aux (f : expr -> expr)
    (Hf : ∀ e : expr, expr_refines (f e) e) (e : expr) {struct e} :
  expr_refines (rewrite_code f e) e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    simpl; try apply expr_refines_refl;
    try (destruct h as [params metadata handler]; simpl);
    try (rewrite_code_cases (rewrite_code_refined_aux f Hf) Hf; fail).
  assert (Hhandlers :
    ∀ vars : env,
      handlers_refine always vars
        (map (fun '(k, h) => (k, rewrite_code_in_handler f h)) handlers)
        handlers).
  { induction handlers as [ | [k [params metadata handler]] handlers IH];
      intros vars; simpl.
    - split; apply Forall2_nil.
    - destruct (IH vars) as [Hforwards Hbackwards].
      split; (apply Forall2_cons; [ | assumption ]);
        (split; [ reflexivity | ]); apply Handler_related;
        intros vs handler_vars conts statics hp rd _ _ _;
        apply (rewrite_code_refined_aux f Hf handler). }
  eapply expr_refines_trans;
    [ | exact (plug_refines (Let_cont_recursive_body _ Hole) _ _ (not_many _)
          (rewrite_code_refined_aux f Hf body)) ].
  intros vars conts statics hp rd.
  destruct (Hhandlers vars) as [Hforwards Hbackwards].
  apply (let_cont_recursive_handlers_refine_where always always_extend);
    [ exact I | apply rewrite_code_handlers_bind_statics
    | exact Hforwards | exact Hbackwards ].
Qed.

Theorem rewrite_code_refines : rewrite_code_refines_statement.
Proof. exact rewrite_code_refines_aux. Qed.

Theorem rewrite_code_equivalent : rewrite_code_equivalent_statement.
Proof.
  intros f Hf e vars conts statics hp rd. apply refines_antisymmetric.
  - apply (rewrite_code_refines_aux f).
    intros e'. apply expr_equivalent_refines, Hf.
  - apply (rewrite_code_refined_aux f).
    intros e' vars' conts' statics' hp' rd'.
    apply equivalent_refines, equivalent_sym, Hf.
Qed.

(* ======================================================================== *)
(* A switch whose arm goes to a switch on the same scrutinee                *)
(* ======================================================================== *)

(* A continuation that is not the one applied may be left out. *)
Lemma apply_cont_skip (ac : apply_cont) (vars : env) (k : continuation)
    (c : cont_value) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  ac_continuation ac <> k ->
  step (State (Apply_cont ac) vars ((k, c) :: conts) statics hp rd)
  = step (State (Apply_cont ac) vars conts statics hp rd).
Proof.
  intros Hk. rewrite !step_apply_cont.
  destruct (eval_simples vars (ac_args ac)); [ | reflexivity ].
  unfold apply_continuation. rewrite (resolve_other _ _ _ _ Hk).
  reflexivity.
Qed.

Lemma thread_arm_fst (k : continuation) (inner : list (Z * apply_cont))
    (i : Z) (ac : apply_cont) :
  fst (thread_arm k inner (i, ac)) = i.
Proof.
  unfold thread_arm. destruct (ac_args ac); [ | reflexivity ].
  destruct (Pos.eqb (ac_continuation ac) k && no_trap_action ac);
    [ | reflexivity ].
  destruct (find_arm i inner) as [ac' | ]; [ | reflexivity ].
  destruct (among k (apply_cont_continuations ac')); reflexivity.
Qed.

Lemma find_arm_thread (k : continuation) (inner : list (Z * apply_cont))
    (i : Z) (arms : list (Z * apply_cont)) :
  find_arm i (map (thread_arm k inner) arms)
  = option_map (fun ac => snd (thread_arm k inner (i, ac))) (find_arm i arms).
Proof.
  induction arms as [ | [d ac] arms IH]; [ reflexivity | ].
  cbn [map find_arm].
  pose proof (thread_arm_fst k inner d ac) as Hfst.
  destruct (thread_arm k inner (d, ac)) as [d' ac'] eqn:Harm.
  simpl in Hfst. subst d'.
  destruct (Z.eqb_spec i d) as [-> | _]; [ | exact IH ].
  cbn [option_map]. rewrite Harm. reflexivity.
Qed.

Lemma thread_arms_equivalent (k : continuation)
    (metadata : handler_metadata) (s : simple)
    (inner arms : list (Z * apply_cont)) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  hm_params_kinds metadata = [] ->
  equivalent
    (State (Switch s arms) vars
       ((k, Closure (Handler [] metadata (Switch s inner)) vars conts [])
        :: conts)
       statics hp rd)
    (State (Switch s (map (thread_arm k inner) arms)) vars
       ((k, Closure (Handler [] metadata (Switch s inner)) vars conts [])
        :: conts)
       statics hp rd).
Proof.
  intros Hno_kinds.
  destruct (eval_simple vars s) as [v | ] eqn:Hs;
    [ | apply step_eq_equivalent; rewrite !step_switch, Hs; reflexivity ].
  destruct (naked_immediate_of_value v) as [i | ] eqn:Hi;
    [ | apply step_eq_equivalent; rewrite !step_switch, Hs, Hi;
        reflexivity ].
  destruct (find_arm i arms) as [ac | ] eqn:Hfind;
    [ | apply step_eq_equivalent;
        rewrite !step_switch, Hs, Hi, find_arm_thread, Hfind; reflexivity ].
  eapply equivalent_trans;
    [ apply step_equivalent; rewrite step_switch, Hs, Hi, Hfind;
      reflexivity
    | ].
  eapply equivalent_trans;
    [ | apply equivalent_sym, step_equivalent;
        rewrite step_switch, Hs, Hi, find_arm_thread, Hfind; reflexivity ].
  unfold thread_arm. destruct ac as [applied args trap]. simpl.
  destruct args; [ | apply equivalent_refl ].
  destruct (Pos.eqb_spec applied k) as [-> | _]; [ | apply equivalent_refl ].
  destruct trap as [action | ]; [ apply equivalent_refl | ].
  unfold no_trap_action. simpl.
  destruct (find_arm i inner) as [ac' | ] eqn:Hinner;
    [ | apply equivalent_refl ].
  match goal with
  | |- context [if ?names then _ else _] =>
    destruct names eqn:Hnames; [ apply equivalent_refl | ]
  end.
  assert (Hother : ac_continuation ac' <> k).
  { apply orb_false_iff in Hnames. destruct Hnames as [Hhead _].
    apply Pos.eqb_neq in Hhead. congruence. }
  simpl.
  apply (equivalent_trans _ (State (Apply_cont ac') vars conts statics hp rd)).
  - eapply equivalent_trans;
      [ apply step_equivalent;
        apply
          (step_apply_cont_handler _ _ _ _ _ rd_ [] [] metadata (Switch s inner)
          vars conts []);
        [ reflexivity | apply resolve_head
        | rewrite Hno_kinds; reflexivity | reflexivity ]
      | ].
    apply step_equivalent. rewrite step_switch, Hs, Hi, Hinner.
    reflexivity.
  - apply equivalent_sym, step_eq_equivalent, apply_cont_skip, Hother.
Qed.

Lemma find_arm_not_applying (k : continuation) (i : Z)
    (arms : list (Z * apply_cont)) (ac : apply_cont) :
  arms_apply k arms = false ->
  find_arm i arms = Some ac -> ac_continuation ac <> k.
Proof.
  unfold arms_apply.
  induction arms as [ | [d ac'] arms IH]; simpl; [ discriminate | ].
  intros H. apply orb_false_iff in H as [Hhead Hrest].
  destruct (Z.eqb i d); [ | exact (IH Hrest) ].
  intros [= <-]. unfold among, apply_cont_continuations in Hhead.
  simpl in Hhead. apply orb_false_iff in Hhead as [Hhead _].
  apply Pos.eqb_neq in Hhead. congruence.
Qed.

Lemma switch_skip (k : continuation) (c : cont_value) (s : simple)
    (arms : list (Z * apply_cont)) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  arms_apply k arms = false ->
  equivalent (State (Switch s arms) vars ((k, c) :: conts) statics hp rd)
    (State (Switch s arms) vars conts statics hp rd).
Proof.
  intros Harms.
  destruct (eval_simple vars s) as [v | ] eqn:Hs;
    [ | apply step_eq_equivalent; rewrite !step_switch, Hs; reflexivity ].
  destruct (naked_immediate_of_value v) as [i | ] eqn:Hi;
    [ | apply step_eq_equivalent; rewrite !step_switch, Hs, Hi;
        reflexivity ].
  destruct (find_arm i arms) as [ac | ] eqn:Hfind;
    [ | apply step_eq_equivalent; rewrite !step_switch, Hs, Hi, Hfind;
        reflexivity ].
  eapply equivalent_step;
    [ rewrite step_switch, Hs, Hi, Hfind; reflexivity
    | rewrite step_switch, Hs, Hi, Hfind; reflexivity | ].
  apply step_eq_equivalent, apply_cont_skip.
  exact (find_arm_not_applying k i arms ac Harms Hfind).
Qed.

Lemma thread_switch_equivalent (k : continuation) (h : continuation_handler)
    (body : expr) :
  expr_equivalent (Let_cont_non_recursive k h body) (thread_switch k h body).
Proof.
  unfold thread_switch.
  destruct h as [[ | param params] metadata handler];
    try (intros vars conts statics hp rd; apply equivalent_refl).
  destruct handler;
    try (intros vars conts statics hp rd; apply equivalent_refl).
  destruct body; try (intros vars conts statics hp rd; apply equivalent_refl).
  destruct (hm_params_kinds metadata) eqn:Hno_kinds;
    [ | intros vars conts statics hp rd; apply equivalent_refl ].
  destruct (simple_eq_dec scrutinee0 scrutinee) as [-> | _];
    [ | intros vars conts statics hp rd; apply equivalent_refl ].
  intros vars conts statics hp rd.
  destruct (arms_apply k (map (thread_arm k arms) arms0)) eqn:Happly.
  - apply let_cont_non_recursive_equivalent_step, thread_arms_equivalent,
      Hno_kinds.
  - eapply equivalent_trans;
      [ apply step_equivalent, step_let_cont_non_recursive | ].
    eapply equivalent_trans; [ apply thread_arms_equivalent, Hno_kinds | ].
    apply switch_skip, Happly.
Qed.

Fixpoint thread_switches_equivalent_aux (e : expr) :
  expr_equivalent e (thread_switches e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try (intros vars conts statics hp rd; apply equivalent_refl).
  - exact (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _)
      (thread_switches_equivalent_aux body)).
  - exact (plug_equivalent (Let_static_body bindings Hole) _ _ (not_many _)
      (thread_switches_equivalent_aux body)).
  - apply (expr_equivalent_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (thread_switches handler)) body)).
    + exact (plug_equivalent
        (Let_cont_non_recursive_handler k params metadata Hole body)
        _ _ (not_many _)
        (thread_switches_equivalent_aux handler)).
    + apply (expr_equivalent_trans _
        (Let_cont_non_recursive k
           (Handler params metadata (thread_switches handler))
           (thread_switches body))).
      * exact (plug_equivalent
          (Let_cont_non_recursive_body k
             (Handler params metadata (thread_switches handler)) Hole)
          _ _ (not_many _) (thread_switches_equivalent_aux body)).
      * apply thread_switch_equivalent.
  - exact (plug_equivalent (Let_cont_recursive_body handlers Hole)
      _ _ (not_many _) (thread_switches_equivalent_aux body)).
Qed.

Theorem thread_switches_equivalent : thread_switches_equivalent_statement.
Proof. exact thread_switches_equivalent_aux. Qed.
