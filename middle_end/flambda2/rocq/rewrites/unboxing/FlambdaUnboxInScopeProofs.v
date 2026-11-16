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


(* Proofs of the statements of FlambdaUnboxInScopeSpec.v. Nothing here
   needs review.

   What is known of symbols alone, where it holds in the static
   environment, holds in the sense of FlambdaUnboxParamsProofs.v under
   any variables ([symbols_facts_hold]); and what holds in that sense,
   made of symbols alone, holds in the static environment
   ([facts_hold_symbols_bound]). So it holds under a group of static
   bindings by the lemma of that file, and in a static environment that
   extends one where it holds. The two rewrites are then proved as
   FlambdaNormaliseSpec.rewrite_code and
   FlambdaRecHandlersSpec.rewrite_rec_handlers are, by the lemmas of
   FlambdaContextProofs.v for a condition on the static environment
   that its extensions keep. For the code of a group, the condition is
   that of the symbols of the group too: it holds of the static
   environment that the group gives, with the code as it was and with
   the code rewritten. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaContextSpec FlambdaContextProofs FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaRecHandlersSpec FlambdaRecHandlersProofs FlambdaBlockLoadSpec
  FlambdaBlockLoadProofs FlambdaUnboxSpec FlambdaUnboxProofs
  FlambdaUnboxParamsSpec FlambdaUnboxParamsProofs FlambdaUnboxVariantsSpec
  FlambdaUnboxVariantsProofs FlambdaUnboxInScopeSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known of symbols                                                 *)
(* ======================================================================== *)

Lemma closed_field_idempotent (field : option simple) :
  closed_field (closed_field field) = closed_field field.
Proof. destruct field as [[x | s | c] | ]; reflexivity. Qed.

Lemma facts_of_symbols_only (known : facts) :
  symbols_only (facts_of_symbols known).
Proof.
  unfold symbols_only, facts_of_symbols.
  cbn [facts_blocks facts_numbers known_symbols known_boxed_symbols].
  do 2 f_equal. rewrite map_map. apply map_ext. intros [s [tag fields]].
  cbn [fst snd]. rewrite map_map. do 2 f_equal.
  apply map_ext, closed_field_idempotent.
Qed.

Theorem symbols_under_statics_only : symbols_under_statics_only_statement.
Proof. intros known bindings. apply facts_of_symbols_only. Qed.

Lemma no_facts_only : symbols_only no_facts.
Proof. reflexivity. Qed.

Lemma no_symbols_bound (statics : static_env) :
  symbols_bound_as no_facts statics.
Proof. split; intros; contradiction. Qed.

(* A field that is as it is known to be with no variables is so under
   any. *)
Lemma field_as_holds (vars : env) (field : option simple) (v : value) :
  field_as field v -> field_holds vars field v.
Proof.
  destruct field as [[x | s | c] | ]; cbn; intros Hfield; try exact Hfield.
  discriminate Hfield.
Qed.

Lemma field_holds_as (vars : env) (field : option simple) (v : value) :
  field_holds vars field v -> field_as (closed_field field) v.
Proof.
  destruct field as [[x | s | c] | ]; cbn; intros Hfield; try exact Hfield.
  exact I.
Qed.

(* What is known of symbols alone, where the static environment binds
   them so, holds under any variables. *)
Lemma symbols_facts_hold (known : facts) (vars : env) (statics : static_env)
    (hp : heap) (rd : read_record) :
  symbols_only known ->
  symbols_bound_as known statics ->
  facts_hold known vars statics hp rd.
Proof.
  intros Honly [Hblocks Hnumbers].
  assert (Hvars : known_vars (facts_blocks known) = []).
  { rewrite <- Honly. reflexivity. }
  assert (Hwrapped : known_wrapped_vars (facts_numbers known) = []).
  { rewrite <- Honly. reflexivity. }
  assert (Hreads : facts_reads known = []).
  { rewrite <- Honly. reflexivity. }
  split; [ split | split; [ split | split; [ | split ] ] ].
  - intros entry Hentry. rewrite Hvars in Hentry. destruct Hentry.
  - intros [s [tag fields]] Hentry.
    destruct (Hblocks s tag fields Hentry) as (vs & Hfields & Hlookup).
    exists vs. split; [ | exact Hlookup ].
    cbn [fst snd]. clear Hlookup Hentry.
    induction Hfields as [ | field v fields vs Hfield _ IH];
      [ apply Forall2_nil | ].
    apply Forall2_cons; [ exact (field_as_holds vars field v Hfield) | ].
    exact IH.
  - intros entry Hentry. rewrite Hwrapped in Hentry. destruct Hentry.
  - intros [s c] Hentry. exact (Hnumbers s c Hentry).
  - intros y block tag vs Hentry. rewrite Hvars in Hentry. destruct Hentry.
  - intros y kind contents c Hentry. rewrite Hwrapped in Hentry.
    destruct Hentry.
  - intros entry Hentry. rewrite Hreads in Hentry. destruct Hentry.
Qed.

(* What holds, made of symbols alone, is how the static environment
   binds the symbols. *)
Lemma facts_hold_symbols_bound (known : facts) (vars : env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  facts_hold known vars statics hp rd ->
  symbols_bound_as (facts_of_symbols known) statics.
Proof.
  intros ([_ Hblocks] & [_ Hnumbers] & _). split.
  - intros s tag fields Hentry.
    cbn [facts_of_symbols facts_blocks known_symbols] in Hentry.
    apply in_map_iff in Hentry as (entry & Heq & Hentry).
    destruct entry as [s' [tag' fields']]. cbn [fst snd] in Heq.
    injection Heq as <- <- <-.
    destruct (Hblocks _ Hentry) as (vs & Hfields & Hlookup).
    exists vs. split; [ | exact Hlookup ].
    cbn [fst snd] in Hfields. clear Hlookup Hentry.
    induction Hfields as [ | field v fields vs Hfield _ IH]; cbn [map];
      [ apply Forall2_nil | ].
    apply Forall2_cons; [ exact (field_holds_as vars field v Hfield) | ].
    exact IH.
  - intros s c Hentry. exact (Hnumbers (s, c) Hentry).
Qed.

Theorem symbols_under_statics_bound : symbols_under_statics_bound_statement.
Proof.
  intros known bindings vars statics extended Honly Hbound Heval.
  apply (facts_hold_symbols_bound _ vars extended empty_heap no_reads).
  exact (facts_hold_under_statics known vars statics bindings extended
    empty_heap no_reads
    (symbols_facts_hold known vars statics empty_heap no_reads Honly Hbound)
    Heval).
Qed.

Lemma symbols_bound_as_extend (known : facts)
    (statics statics' : static_env) :
  symbols_bound_as known statics ->
  statics_extend statics statics' ->
  symbols_bound_as known statics'.
Proof.
  intros [Hblocks Hnumbers] [_ Hvalues]. split.
  - intros s tag fields Hentry.
    destruct (Hblocks s tag fields Hentry) as (vs & Hfields & Hlookup).
    exists vs. split; [ exact Hfields | exact (Hvalues _ _ Hlookup) ].
  - intros s c Hentry. exact (Hvalues _ _ (Hnumbers s c Hentry)).
Qed.

(* The symbols of a group are those of the group with its code
   rewritten. *)
Lemma blocks_of_rewritten_code (g : expr -> expr)
    (bindings : list static_binding) :
  flat_map static_block_made (map (rewrite_code_in_binding g) bindings)
  = flat_map static_block_made bindings.
Proof.
  induction bindings as [ | binding bindings IH]; [ reflexivity | ].
  cbn [map flat_map]. rewrite IH.
  destruct binding
    as [id [params my_closure ret exn metadata body] | id | s const];
    reflexivity.
Qed.

Lemma numbers_of_rewritten_code (g : expr -> expr)
    (bindings : list static_binding) :
  flat_map static_number_boxed (map (rewrite_code_in_binding g) bindings)
  = flat_map static_number_boxed bindings.
Proof.
  induction bindings as [ | binding bindings IH]; [ reflexivity | ].
  cbn [map flat_map]. rewrite IH.
  destruct binding
    as [id [params my_closure ret exn metadata body] | id | s const];
    reflexivity.
Qed.

Lemma symbols_under_rewritten_code (g : expr -> expr) (known : facts)
    (bindings : list static_binding) :
  symbols_under_statics known (map (rewrite_code_in_binding g) bindings)
  = symbols_under_statics known bindings.
Proof.
  unfold symbols_under_statics, facts_under_statics, known_under_statics,
    numbers_under_statics.
  rewrite blocks_of_rewritten_code, numbers_of_rewritten_code. reflexivity.
Qed.

(* ======================================================================== *)
(* The unboxing given what is known of symbols                              *)
(* ======================================================================== *)

Theorem unbox_variants_knowing_symbols_refines :
  unbox_variants_knowing_symbols_refines_statement.
Proof.
  intros known e vars conts statics hp rd Honly Hbound.
  apply unbox_variants_knowing_refines.
  exact (symbols_facts_hold known vars statics hp rd Honly Hbound).
Qed.

(* ======================================================================== *)
(* The two rewrites                                                         *)
(* ======================================================================== *)

Section In_scope.

Variable f : facts -> expr -> expr.

Hypothesis f_refines : refines_where_known f.

(* What [rewrite_code_body] gives refines where what is known holds. *)
Lemma rewritten_body_refines (known : facts) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  symbols_only known ->
  symbols_bound_as known statics ->
  refines (State body vars conts statics hp rd)
    (State (rewrite_code_body (f known) body) vars conts statics hp rd).
Proof.
  intros Honly Hbound. unfold rewrite_code_body. cbv zeta.
  destruct (Bool.eqb (binds_statics body) (binds_statics (f known body)));
    [ apply f_refines; assumption | apply refines_refl ].
Qed.

(* The code of a group refines its counterparts where what is known
   holds. *)
Lemma rewritten_bindings_refine (known : facts)
    (bindings : list static_binding) :
  symbols_only known ->
  bindings_refine (symbols_bound_as known) bindings
    (map (rewrite_code_in_binding (f known)) bindings).
Proof.
  intros Honly.
  induction bindings
    as [ | [id [params my_closure ret exn metadata code_body] | id
           | sym const]
           bindings [Hforwards Hbackwards]];
    simpl.
  - split; apply Forall2_nil.
  - unfold rewrite_code_body.
    destruct (Bool.eqb (binds_statics code_body)
                (binds_statics (f known code_body))) eqn:Hbinds;
      [ apply eqb_prop in Hbinds | ];
      split; (apply Forall2_cons; [ | assumption ]);
      try apply Binding_same;
      apply Binding_code, Codes_related;
      first
        [ exact Hbinds | exact (eq_sym Hbinds)
        | intros vars conts statics hp rd Hbound; apply f_refines;
          assumption ].
  - split; (apply Forall2_cons; [ apply Binding_same | assumption ]).
  - split; (apply Forall2_cons; [ apply Binding_same | assumption ]).
Qed.

Fixpoint rewrite_code_in_scope_binds_statics (e : expr) :
  ∀ known : facts,
    binds_statics (rewrite_code_in_scope f known e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros known;
    cbn [rewrite_code_in_scope rewrite_code_in_scope_in_handler
         binds_statics];
    cbv zeta; try reflexivity.
  - apply rewrite_code_in_scope_binds_statics.
  - rewrite rewrite_code_in_scope_binds_statics. f_equal.
    generalize (f (symbols_under_statics known bindings)). intros g.
    induction bindings
      as [ | [id [params my_closure ret exn metadata code_body] | id
             | sym const]
             bindings IH];
      simpl; congruence.
  - destruct h as [params metadata handler].
    cbn [rewrite_code_in_scope_in_handler handler_binds_statics].
    rewrite !rewrite_code_in_scope_binds_statics. reflexivity.
  - rewrite rewrite_code_in_scope_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite rewrite_code_in_scope_binds_statics, IH. reflexivity.
Qed.

Lemma rewrite_code_in_scope_handlers_bind_statics (known : facts)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map (fun '(k, h) => (k, rewrite_code_in_scope_in_handler f known h))
       handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
    [ reflexivity | ].
  rewrite rewrite_code_in_scope_binds_statics, IH. reflexivity.
Qed.

(* Down the term, with the static environment binding the symbols as
   they are known to be: at once in a body, and from then on in a
   handler and in code. *)
Fixpoint rewrite_code_in_scope_refines_aux (e : expr) :
  ∀ (known : facts) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record),
    symbols_only known ->
    symbols_bound_as known statics ->
    refines (State e vars conts statics hp rd)
      (State (rewrite_code_in_scope f known e) vars conts statics hp rd).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros known vars conts statics hp rd Honly Hbound;
    cbn [rewrite_code_in_scope]; cbv zeta; try apply refines_refl.
  - apply let_refines_step_exec; [ reflexivity | intros v hp' rd' _ _ ].
    apply rewrite_code_in_scope_refines_aux; assumption.
  - (* the code of the group is rewritten with the symbols of the group
       known: the body under the one static environment refines it
       under the other; then down the body *)
    set (known' := symbols_under_statics known bindings).
    assert (Honly' : symbols_only known') by apply facts_of_symbols_only.
    destruct (rewritten_bindings_refine known' bindings Honly')
      as [Hforwards Hbackwards].
    apply (fun H =>
      eval_static_bindings_related _ vars _ _ H statics statics
        (statics_related_refl _ statics)) in Hforwards, Hbackwards.
    apply let_static_refines_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Hbindings,
      (eval_static_bindings vars
         (map (rewrite_code_in_binding (f known')) bindings) statics)
      as [extended' | ] eqn:Hbindings';
      try contradiction; [ | exact I ].
    assert (Hbound' : symbols_bound_as known' extended).
    { exact (symbols_under_statics_bound known bindings vars statics
        extended Honly Hbound Hbindings). }
    assert (Hbound'' : symbols_bound_as known' extended').
    { unfold known' at 1.
      rewrite <- (symbols_under_rewritten_code (f known') known bindings).
      exact (symbols_under_statics_bound known _ vars statics extended'
        Honly Hbound Hbindings'). }
    apply (refines_trans _ (State body vars conts extended' hp rd)).
    + apply (related_states_refines (symbols_bound_as known')
        (symbols_bound_as_extend known'));
        [ exact Hbound' | exact Hbound'' | apply cont_env_related_refl
        | apply cont_env_related_refl | exact Hforwards | exact Hbackwards ].
    + apply rewrite_code_in_scope_refines_aux; assumption.
  - destruct h as [params metadata handler].
    cbn [rewrite_code_in_scope_in_handler].
    apply (refines_trans _
      (State
         (Let_cont_non_recursive k
            (Handler params metadata (rewrite_code_in_scope f known handler))
            body)
         vars conts statics hp rd)).
    + apply (let_cont_handler_refines_where (symbols_bound_as known)
        (symbols_bound_as_extend known)); [ exact Hbound | ].
      intros vs handler_vars conts' statics' hp' rd' Hbound' _ _.
      apply rewrite_code_in_scope_refines_aux; assumption.
    + apply let_cont_non_recursive_refines_step,
        rewrite_code_in_scope_refines_aux; assumption.
  - apply (refines_trans _
      (State
         (Let_cont_recursive
            (map
               (fun '(k, h) =>
                  (k, rewrite_code_in_scope_in_handler f known h))
               handlers)
            body)
         vars conts statics hp rd)).
    + assert (Hhandlers :
        handlers_refine (symbols_bound_as known) vars handlers
          (map
             (fun '(k, h) => (k, rewrite_code_in_scope_in_handler f known h))
             handlers)).
      { induction handlers
          as [ | [k [params metadata handler]] handlers IH];
          simpl.
        - split; apply Forall2_nil.
        - destruct IH as [Hforwards Hbackwards].
          split; (apply Forall2_cons; [ | assumption ]);
            (split; [ reflexivity | ]); apply Handler_related;
            intros vs handler_vars conts' statics' hp' rd' Hbound' _ _;
            apply rewrite_code_in_scope_refines_aux; assumption. }
      destruct Hhandlers as [Hforwards Hbackwards].
      apply (let_cont_recursive_handlers_refine_where
        (symbols_bound_as known) (symbols_bound_as_extend known));
        try assumption.
      symmetry. apply rewrite_code_in_scope_handlers_bind_statics.
    + apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
      apply rewrite_code_in_scope_refines_aux; assumption.
Qed.

Fixpoint rewrite_rec_handlers_in_scope_binds_statics (e : expr) :
  ∀ known : facts,
    binds_statics (rewrite_rec_handlers_in_scope f known e)
    = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros known;
    cbn [rewrite_rec_handlers_in_scope
         rewrite_rec_handlers_in_scope_in_handler binds_statics];
    try reflexivity.
  - apply rewrite_rec_handlers_in_scope_binds_statics.
  - rewrite rewrite_rec_handlers_in_scope_binds_statics. reflexivity.
  - destruct h as [params metadata handler].
    cbn [rewrite_rec_handlers_in_scope_in_handler handler_binds_statics].
    rewrite !rewrite_rec_handlers_in_scope_binds_statics. reflexivity.
  - rewrite rewrite_rec_handlers_in_scope_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite rewrite_code_body_binds_statics,
      rewrite_rec_handlers_in_scope_binds_statics, IH.
    reflexivity.
Qed.

Lemma rewrite_rec_handlers_in_scope_bind_statics (known : facts)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map (fun '(k, h) => (k, rewrite_rec_handler_in_scope f known h))
       handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
    [ reflexivity | ].
  rewrite rewrite_code_body_binds_statics,
    rewrite_rec_handlers_in_scope_binds_statics, IH.
  reflexivity.
Qed.

Fixpoint rewrite_rec_handlers_in_scope_refines_aux (e : expr) :
  ∀ (known : facts) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record),
    symbols_only known ->
    symbols_bound_as known statics ->
    refines (State e vars conts statics hp rd)
      (State (rewrite_rec_handlers_in_scope f known e) vars conts statics
         hp rd).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros known vars conts statics hp rd Honly Hbound;
    cbn [rewrite_rec_handlers_in_scope]; try apply refines_refl.
  - apply let_refines_step_exec; [ reflexivity | intros v hp' rd' _ _ ].
    apply rewrite_rec_handlers_in_scope_refines_aux; assumption.
  - apply let_static_refines_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Hbindings; [ | exact I ].
    apply rewrite_rec_handlers_in_scope_refines_aux;
      [ apply facts_of_symbols_only | ].
    exact (symbols_under_statics_bound known bindings vars statics extended
      Honly Hbound Hbindings).
  - destruct h as [params metadata handler].
    cbn [rewrite_rec_handlers_in_scope_in_handler].
    apply (refines_trans _
      (State
         (Let_cont_non_recursive k
            (Handler params metadata
               (rewrite_rec_handlers_in_scope f known handler))
            body)
         vars conts statics hp rd)).
    + apply (let_cont_handler_refines_where (symbols_bound_as known)
        (symbols_bound_as_extend known)); [ exact Hbound | ].
      intros vs handler_vars conts' statics' hp' rd' Hbound' _ _.
      apply rewrite_rec_handlers_in_scope_refines_aux; assumption.
    + apply let_cont_non_recursive_refines_step,
        rewrite_rec_handlers_in_scope_refines_aux; assumption.
  - apply (refines_trans _
      (State
         (Let_cont_recursive
            (map
               (fun '(k, h) => (k, rewrite_rec_handler_in_scope f known h))
               handlers)
            body)
         vars conts statics hp rd)).
    + assert (Hhandlers :
        handlers_refine (symbols_bound_as known) vars handlers
          (map (fun '(k, h) => (k, rewrite_rec_handler_in_scope f known h))
             handlers)).
      { induction handlers
          as [ | [k [params metadata handler]] handlers IH];
          simpl.
        - split; apply Forall2_nil.
        - destruct IH as [Hforwards Hbackwards].
          split; (apply Forall2_cons; [ | assumption ]);
            (split; [ reflexivity | ]); apply Handler_related;
            intros vs handler_vars conts' statics' hp' rd' Hbound' _ _;
            (eapply refines_trans;
              [ apply rewrite_rec_handlers_in_scope_refines_aux
              | apply rewritten_body_refines ]);
            assumption. }
      destruct Hhandlers as [Hforwards Hbackwards].
      apply (let_cont_recursive_handlers_refine_where
        (symbols_bound_as known) (symbols_bound_as_extend known));
        try assumption.
      symmetry. apply rewrite_rec_handlers_in_scope_bind_statics.
    + apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
      apply rewrite_rec_handlers_in_scope_refines_aux; assumption.
Qed.

End In_scope.

Theorem rewrite_code_in_scope_refines :
  rewrite_code_in_scope_refines_statement.
Proof.
  intros f Hf known e vars conts statics hp Honly Hbound.
  exact (rewrite_code_in_scope_refines_aux f Hf e known vars conts statics
    hp Honly Hbound).
Qed.

Theorem rewrite_rec_handlers_in_scope_refines :
  rewrite_rec_handlers_in_scope_refines_statement.
Proof.
  intros f Hf known e vars conts statics hp Honly Hbound.
  exact (rewrite_rec_handlers_in_scope_refines_aux f Hf e known vars conts
    statics hp Honly Hbound).
Qed.

Theorem rewrite_code_in_scope_expr_refines :
  rewrite_code_in_scope_expr_refines_statement.
Proof.
  intros f e Hf vars conts statics hp rd.
  exact (rewrite_code_in_scope_refines f Hf no_facts e vars conts statics hp
    rd_ no_facts_only (no_symbols_bound statics)).
Qed.

Theorem rewrite_rec_handlers_in_scope_expr_refines :
  rewrite_rec_handlers_in_scope_expr_refines_statement.
Proof.
  intros f e Hf vars conts statics hp rd.
  exact (rewrite_rec_handlers_in_scope_refines f Hf no_facts e vars conts
    statics hp rd_ no_facts_only (no_symbols_bound statics)).
Qed.
