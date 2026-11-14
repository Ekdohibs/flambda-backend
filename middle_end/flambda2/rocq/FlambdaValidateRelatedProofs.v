(* Proofs of the statements of FlambdaValidateRelatedSpec.v. Nothing
   here needs review.

   The check is followed step by step, as
   FlambdaValidateImportsProofs.validate_unit_sound_lifted_imports
   follows [validate_unit], with the two terms run from one
   continuation environment of what may follow a unit. A step that is
   a refinement or an equivalence under all environments is taken at
   that one. The others have a statement from what may follow a unit:
   the liftings (FlambdaLiftObservedSpec.v), the use of what the unit
   binds its symbols to (FlambdaKnownSymbolsSpec.v), the reading of
   known value slots (FlambdaKnownSlotsSpec.v), the removal of value
   slots (FlambdaUnusedSlotsObservedSpec.v), the code and the static
   bindings bound first (FlambdaHoistCodeSpec.v,
   FlambdaHoistStaticsObservedSpec.v); and the removals of static
   bindings hold under any environments in which the names removed do
   not occur (FlambdaUnusedStaticsSpec.v,
   FlambdaUnusedHeadStaticsSpec.v), which what may follow a unit is
   (FlambdaLiftObservedProofs.observers_free_of). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaBindersSpec FlambdaCompareSpec
  FlambdaCompareProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaStaticsOrderSpec FlambdaStaticsOrderProofs FlambdaUnusedStaticsSpec
  FlambdaUnusedStaticsProofs FlambdaUnusedHeadStaticsSpec
  FlambdaUnusedHeadStaticsProofs FlambdaLoopifySpec FlambdaLoopifyProofs
  FlambdaLiftSpec FlambdaKnownSymbolsSpec FlambdaKnownSymbolsProofs
  FlambdaKnownClosuresSpec FlambdaKnownClosuresProofs FlambdaKnownSlotsSpec
  FlambdaKnownSlotsProofs FlambdaFloatCodeSpec FlambdaFloatCodeProofs
  FlambdaHoistCodeSpec
  FlambdaHoistCodeProofs FlambdaGuardScopeSpec FlambdaGuardScopeProofs
  FlambdaValidateSpec FlambdaValidateProofs FlambdaImportsSpec
  FlambdaImportsProofs FlambdaValidateImportsSpec
  FlambdaValidateImportsProofs FlambdaStubCallsSpec FlambdaStubCallsProofs
  FlambdaNeverReturnsSpec FlambdaNeverReturnsProofs FlambdaExitWorldSpec
  FlambdaRunUnderSpec FlambdaObserversSpec FlambdaObserversProofs
  FlambdaLiftObservedSpec FlambdaLiftObservedProofs FlambdaUnitFromProofs
  FlambdaUnusedSlotsSpec FlambdaUnusedSlotsProofs
  FlambdaUnusedSlotsObservedSpec FlambdaUnusedSlotsObservedProofs
  FlambdaHoistStaticsSpec FlambdaHoistStaticsObservedSpec
  FlambdaHoistStaticsObservedProofs FlambdaValidateRelatedSpec.
From OxCamlFlambda2 Require FlambdaSharedReadsProofs.
From OxCamlFlambda2 Require FlambdaClosureOfCodeProofs.
From OxCamlFlambda2Types Require FlambdaSimplifyCarriedSpec
  FlambdaSimplifyCarriedProofs.
Import ListNotations.

(* A term projects none of no value slots: what may follow a unit and
   projects none of the slots given is what may follow a unit. *)
Fixpoint slots_unused_none (e : expr) {struct e} : slots_unused [] e = true
with handler_slots_unused_none (h : continuation_handler) {struct h} :
  handler_slots_unused [] h = true
with binding_slots_unused_none (binding : static_binding) {struct binding} :
  binding_slots_unused [] binding = true
with code_slots_unused_none (c : code) {struct c} :
  code_slots_unused [] c = true.
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + rewrite slots_unused_none. destruct d; reflexivity.
    + rewrite slots_unused_none, andb_true_r.
      induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
      rewrite binding_slots_unused_none, IH. reflexivity.
    + rewrite handler_slots_unused_none, slots_unused_none. reflexivity.
    + rewrite slots_unused_none, andb_true_r.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite handler_slots_unused_none, IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply slots_unused_none.
  - destruct binding as [id c | id | sym const]; simpl; try reflexivity.
    apply code_slots_unused_none.
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    apply slots_unused_none.
Qed.

Lemma observers_any_slots (slots : list value_slot) (conts : cont_env) :
  observers slots conts -> observers [] conts.
Proof.
  unfold observers. apply Forall_impl. intros [k c]. simpl.
  destruct c as [h vars cs recursive | k' | c extra | remaining ra rr rt et];
    try (intros H; exact H).
  destruct h as [params metadata body].
  destruct vars; try (intros H; exact H).
  destruct cs; try (intros H; exact H).
  destruct recursive; try (intros H; exact H).
  intros (Hbinds & Hscope & _).
  split; [ exact Hbinds | ]. split; [ exact Hscope | ].
  apply slots_unused_none.
Qed.

Section Followed.

Variable imports : list (symbol * import_class).
Variable slots : list value_slot.
Variable conts : cont_env.
Hypothesis Hfollowed : observers slots conts.

Let statics := imports_statics imports.

Let Hobservers : observers [] conts :=
  observers_any_slots slots conts Hfollowed.

(* The two terms, each followed by [conts] and run from the imports,
   come to results that differ in nothing that an exit is compared
   by. *)
Definition reaches (s t : expr) : Prop :=
  observed_refines (Step (State s [] conts statics empty_heap))
    (Step (State t [] conts statics empty_heap)).

Lemma reaches_trans (s t u : expr) : reaches s t -> reaches t u -> reaches s u.
Proof. apply observed_refines_trans. Qed.

Lemma refines_reaches (s t : expr) :
  refines (State s [] conts statics empty_heap)
    (State t [] conts statics empty_heap) ->
  reaches s t.
Proof. apply refines_observed. Qed.

Lemma envs_free_of_followed (D : dropped) :
  envs_free_of D [] conts statics = true.
Proof.
  unfold envs_free_of.
  rewrite (observers_free_of D slots conts Hfollowed).
  unfold statics. rewrite (imports_statics_free D imports). reflexivity.
Qed.

(* ------------------------------------------------------------------------ *)
(* The steps                                                                *)
(* ------------------------------------------------------------------------ *)

Lemma with_lifting_observed_reaches (L : lifting) (e : expr) :
  lifting_observed_free_in statics L e = true ->
  reaches e (with_lifting_observed L e).
Proof.
  unfold lifting_observed_free_in. cbv zeta. intros H.
  apply andb_true_iff in H as [H H4].
  apply andb_true_iff in H as [H H3].
  apply andb_true_iff in H as [H1 H2].
  pose proof (with_lifting_observed_refines imports L e) as Hlifting.
  cbv zeta in Hlifting.
  exact (Hlifting H1 H2 H3 H4 conts Hobservers).
Qed.

Lemma iterate_lifted_refines_followed (rounds : nat)
    (avoid : list continuation) (kept : list code_id) :
  ∀ e : expr,
    known_free_in statics rounds avoid kept e = true ->
    refines (State e [] conts statics empty_heap)
      (State (iterate rounds (normalise_lifted_round avoid kept) e) [] conts
         statics empty_heap).
Proof.
  induction rounds as [ | rounds IH]; intros e;
    cbn [known_free_in iterate];
    [ intros _; apply refines_refl | ].
  cbv zeta.
  set (given :=
    use_known_closures (normalise_before_round_avoiding avoid kept e)).
  intros Hfree.
  (* One round, and what the test says after it: nothing where the
     round gives back the term it was given, which is where [iterate]
     stops, and the test of the other rounds otherwise. *)
  assert (Hround :
    refines (State e [] conts statics empty_heap)
      (State (normalise_lifted_round avoid kept e) [] conts statics
         empty_heap)
    ∧ (if expr_eqb [] [] e (normalise_lifted_round avoid kept e) then true
       else
         known_free_in statics rounds avoid kept
           (normalise_lifted_round avoid kept e))
      = true).
  { assert (Hgiven_refines :
      refines (State e [] conts statics empty_heap)
        (State given [] conts statics empty_heap)).
    { apply (refines_trans _ _ _
        (normalise_before_round_avoiding_refines avoid kept e [] conts
           statics empty_heap)).
      exact (use_known_closures_refines _ [] conts statics empty_heap). }
    unfold normalise_lifted_round. fold given. unfold with_known.
    destruct (fold_known_unit given) as [folded | ] eqn:Hfold.
    - apply andb_true_iff in Hfree as [Hgiven Hfree].
      split; [ | exact Hfree ].
      apply (refines_trans _ _ _ Hgiven_refines).
      apply (refines_trans _ _ _
        (fold_known_unit_refines_observed imports conts given folded Hfold
           Hgiven Hobservers)).
      apply (refines_trans _ _ _
        (read_known_slots_refines_observed imports conts folded
           Hobservers)).
      exact (FlambdaSharedReadsProofs.share_static_reads_refines_observed
               imports conts _ Hobservers).
    - split; [ | exact Hfree ].
      apply (refines_trans _ _ _ Hgiven_refines).
      apply (refines_trans _ _ _
        (read_known_slots_refines_observed imports conts given
           Hobservers)).
      exact (FlambdaSharedReadsProofs.share_static_reads_refines_observed
               imports conts _ Hobservers). }
  destruct Hround as [Hround Hrest].
  destruct (expr_eqb [] [] e (normalise_lifted_round avoid kept e));
    [ exact Hround | ].
  exact (refines_trans _ _ _ Hround (IH _ Hrest)).
Qed.

Lemma normalise_lifted_refines_followed (rounds : nat)
    (avoid : list continuation) (kept : list code_id) (e : expr) :
  normalise_lifted_free_in statics rounds avoid kept e = true ->
  refines (State e [] conts statics empty_heap)
    (State (normalise_lifted rounds avoid kept e) [] conts statics
       empty_heap).
Proof.
  unfold normalise_lifted, normalise_lifted_free_in. cbv zeta.
  intros Hknown.
  apply andb_true_iff in Hknown as [Hfirst Hsecond].
  apply (refines_trans _ _ _
    (iterate_refines _ first_phase_round_refines rounds _ [] conts
       statics empty_heap)).
  apply (refines_trans _ _ _
    (iterate_lifted_refines_followed rounds avoid kept _ Hfirst)).
  apply (refines_trans _ _ _
    (guard_scope_refines _ _
       (FlambdaSimplifyCarriedProofs.simplify_by_types_carried_refines _)
       [] conts statics empty_heap)).
  exact (iterate_lifted_refines_followed rounds avoid kept _ Hsecond).
Qed.

(* One round of [normalise_lifted], from what is proved of its rounds
   in turn: [iterate] of one round is that round. *)
Lemma normalise_lifted_round_refines_followed (avoid : list continuation)
    (kept : list code_id) (e : expr) :
  known_free_in statics 1 avoid kept e = true ->
  refines (State e [] conts statics empty_heap)
    (State (normalise_lifted_round avoid kept e) [] conts statics
       empty_heap).
Proof.
  intros Hfree.
  generalize (iterate_lifted_refines_followed 1 avoid kept e Hfree).
  cbn [iterate]. cbv zeta.
  destruct (expr_eqb [] [] e (normalise_lifted_round avoid kept e));
    exact (fun H => H).
Qed.

(* The same of the rounds that are followed by the calls of the
   closures that a [Let] makes made direct, where they are asked to. *)
Lemma iterate_lifted_calling_refines_followed (calling : bool)
    (rounds : nat) (avoid : list continuation) (kept : list code_id) :
  ∀ e : expr,
    known_calling_free_in statics calling rounds avoid kept e = true ->
    refines (State e [] conts statics empty_heap)
      (State
         (iterate rounds (normalise_lifted_round_calling calling avoid kept)
            e)
         [] conts statics empty_heap).
Proof.
  induction rounds as [ | rounds IH]; intros e;
    cbn [known_calling_free_in iterate];
    [ intros _; apply refines_refl | ].
  cbv zeta. intros Hfree.
  apply andb_true_iff in Hfree as [Hone Hrest].
  assert (Hround :
    refines (State e [] conts statics empty_heap)
      (State (normalise_lifted_round_calling calling avoid kept e) [] conts
         statics empty_heap)).
  { unfold normalise_lifted_round_calling. cbv zeta.
    apply (refines_trans _ _ _
      (normalise_lifted_round_refines_followed avoid kept e Hone)).
    destruct calling; [ | apply refines_refl ].
    exact (with_made_closures_called_refines _ [] conts statics
             empty_heap). }
  destruct
    (expr_eqb [] [] e (normalise_lifted_round_calling calling avoid kept e));
    [ exact Hround | ].
  exact (refines_trans _ _ _ Hround (IH _ Hrest)).
Qed.

Lemma normalise_lifted_calling_refines_followed (calling : bool)
    (rounds : nat) (avoid : list continuation) (kept : list code_id)
    (e : expr) :
  normalise_lifted_calling_free_in statics calling rounds avoid kept e
  = true ->
  refines (State e [] conts statics empty_heap)
    (State (normalise_lifted_calling calling rounds avoid kept e) [] conts
       statics empty_heap).
Proof.
  unfold normalise_lifted_calling, normalise_lifted_calling_free_in.
  cbv zeta. intros Hknown.
  apply andb_true_iff in Hknown as [Hfirst Hsecond].
  apply (refines_trans _ _ _
    (iterate_refines _ first_phase_round_refines rounds _ [] conts
       statics empty_heap)).
  apply (refines_trans _ _ _
    (iterate_lifted_calling_refines_followed calling rounds avoid kept _
       Hfirst)).
  apply (refines_trans _ _ _
    (guard_scope_refines _ _
       (FlambdaSimplifyCarriedProofs.simplify_by_types_carried_refines _)
       [] conts statics empty_heap)).
  exact (iterate_lifted_calling_refines_followed calling rounds avoid kept _
           Hsecond).
Qed.

(* One round of [normalise_lifted_calling], from what is proved of its
   rounds in turn: the test of one such round is that of one round of
   [normalise_lifted]. *)
Lemma normalise_lifted_round_calling_refines_followed (calling : bool)
    (avoid : list continuation) (kept : list code_id) (e : expr) :
  known_free_in statics 1 avoid kept e = true ->
  refines (State e [] conts statics empty_heap)
    (State (normalise_lifted_round_calling calling avoid kept e) [] conts
       statics empty_heap).
Proof.
  intros Hfree.
  assert (Hrounds :
    known_calling_free_in statics calling 1 avoid kept e = true).
  { cbn [known_calling_free_in]. cbv zeta. rewrite Hfree.
    destruct
      (expr_eqb [] [] e
         (normalise_lifted_round_calling calling avoid kept e));
      reflexivity. }
  generalize
    (iterate_lifted_calling_refines_followed calling 1 avoid kept e Hrounds).
  cbn [iterate]. cbv zeta.
  destruct
    (expr_eqb [] [] e (normalise_lifted_round_calling calling avoid kept e));
    exact (fun H => H).
Qed.

(* The same of the rounds that end with what the closures of the code
   of the unit are known to be, where they are asked to: that step
   refines the unit followed by what may follow a unit. *)
Lemma iterate_lifted_own_refines_followed (calling own : bool)
    (rounds : nat) (avoid : list continuation) (kept : list code_id) :
  ∀ e : expr,
    known_own_free_in statics calling own rounds avoid kept e = true ->
    refines (State e [] conts statics empty_heap)
      (State
         (iterate rounds (normalise_lifted_round_own calling own avoid kept)
            e)
         [] conts statics empty_heap).
Proof.
  induction rounds as [ | rounds IH]; intros e;
    cbn [known_own_free_in iterate];
    [ intros _; apply refines_refl | ].
  cbv zeta. intros Hfree.
  apply andb_true_iff in Hfree as [Hone Hrest].
  assert (Hround :
    refines (State e [] conts statics empty_heap)
      (State (normalise_lifted_round_own calling own avoid kept e) [] conts
         statics empty_heap)).
  { unfold normalise_lifted_round_own. cbv zeta.
    apply (refines_trans _ _ _
      (normalise_lifted_round_calling_refines_followed calling avoid kept e
         Hone)).
    destruct own; [ | apply refines_refl ].
    exact (FlambdaClosureOfCodeProofs.know_own_closures_refines_observed
             imports [] conts _ Hobservers). }
  destruct
    (expr_eqb [] [] e (normalise_lifted_round_own calling own avoid kept e));
    [ exact Hround | ].
  exact (refines_trans _ _ _ Hround (IH _ Hrest)).
Qed.

Lemma normalise_lifted_own_refines_followed (calling own : bool)
    (rounds : nat) (avoid : list continuation) (kept : list code_id)
    (e : expr) :
  normalise_lifted_own_free_in statics calling own rounds avoid kept e
  = true ->
  refines (State e [] conts statics empty_heap)
    (State (normalise_lifted_own calling own rounds avoid kept e) [] conts
       statics empty_heap).
Proof.
  unfold normalise_lifted_own, normalise_lifted_own_free_in.
  cbv zeta. intros Hknown.
  apply andb_true_iff in Hknown as [Hfirst Hsecond].
  apply (refines_trans _ _ _
    (iterate_refines _ first_phase_round_refines rounds _ [] conts
       statics empty_heap)).
  apply (refines_trans _ _ _
    (iterate_lifted_own_refines_followed calling own rounds avoid kept _
       Hfirst)).
  apply (refines_trans _ _ _
    (guard_scope_refines _ _
       (FlambdaSimplifyCarriedProofs.simplify_by_types_carried_refines _)
       [] conts statics empty_heap)).
  exact (iterate_lifted_own_refines_followed calling own rounds avoid kept _
           Hsecond).
Qed.

(* The rounds that use what is known after the demotion. *)
Lemma tidy_known_refines_followed (inner : nat) :
  ∀ (rounds : nat) (e : expr),
    tidy_known_free_in statics inner rounds e = true ->
    refines (State e [] conts statics empty_heap)
      (State (iterate rounds (demote_and_tidy_known_round inner) e) [] conts
         statics empty_heap).
Proof.
  induction rounds as [ | rounds IH]; intros e;
    cbn [tidy_known_free_in iterate];
    [ intros _; apply refines_refl | ].
  cbv zeta.
  set (given := use_known_closures e).
  intros Hfree.
  assert (Hround :
    refines (State e [] conts statics empty_heap)
      (State (demote_and_tidy_known_round inner e) [] conts statics
         empty_heap)
    ∧ (if expr_eqb [] [] e (demote_and_tidy_known_round inner e) then true
       else
         tidy_known_free_in statics inner rounds
           (demote_and_tidy_known_round inner e))
      = true).
  { assert (Hgiven_refines :
      refines (State e [] conts statics empty_heap)
        (State given [] conts statics empty_heap)).
    { exact (use_known_closures_refines _ [] conts statics empty_heap). }
    unfold demote_and_tidy_known_round. fold given. unfold with_known.
    destruct (fold_known_unit given) as [folded | ] eqn:Hfold.
    - apply andb_true_iff in Hfree as [Hgiven Hfree].
      split; [ | exact Hfree ].
      apply (refines_trans _ _ _ Hgiven_refines).
      apply (refines_trans _ _ _
        (fold_known_unit_refines_observed imports conts given folded Hfold
           Hgiven Hobservers)).
      apply (refines_trans _ _ _
        (read_known_slots_refines_observed imports conts folded
           Hobservers)).
      apply (refines_trans _ _ _
        (FlambdaSharedReadsProofs.share_static_reads_refines_observed
           imports conts _ Hobservers)).
      exact (demote_and_tidy_refines inner _ [] conts statics empty_heap).
    - split; [ | exact Hfree ].
      apply (refines_trans _ _ _ Hgiven_refines).
      apply (refines_trans _ _ _
        (read_known_slots_refines_observed imports conts given
           Hobservers)).
      apply (refines_trans _ _ _
        (FlambdaSharedReadsProofs.share_static_reads_refines_observed
           imports conts _ Hobservers)).
      exact (demote_and_tidy_refines inner _ [] conts statics empty_heap). }
  destruct Hround as [Hround Hrest].
  destruct (expr_eqb [] [] e (demote_and_tidy_known_round inner e));
    [ exact Hround | ].
  exact (refines_trans _ _ _ Hround (IH _ Hrest)).
Qed.

Lemma demote_and_tidy_known_refines_followed (rounds : nat) (e : expr) :
  demote_and_tidy_known_free_in statics rounds e = true ->
  refines (State e [] conts statics empty_heap)
    (State (demote_and_tidy_known rounds e) [] conts statics empty_heap).
Proof.
  unfold demote_and_tidy_known_free_in, demote_and_tidy_known. intros Hfree.
  apply (refines_trans _ _ _
    (demote_and_tidy_refines rounds e [] conts statics empty_heap)).
  exact (tidy_known_refines_followed rounds rounds _ Hfree).
Qed.

Lemma lifted_in_turn_observed_reaches (rounds : nat)
    (avoid : list continuation) (Ls : list lifting) (kept : list code_id) :
  ∀ e : expr,
    lifted_observed_free_in statics rounds avoid Ls kept e = true ->
    reaches e (lifted_in_turn_observed rounds avoid Ls kept e).
Proof.
  induction Ls as [ | L Ls IH]; intros e;
    cbn [lifted_observed_free_in lifted_in_turn_observed].
  - intros Hfree. apply refines_reaches.
    apply (refines_trans _ _ _
      (read_known_slots_anywhere_refines_observed imports conts e
         Hobservers)).
    exact (demote_and_tidy_known_refines_followed rounds _ Hfree).
  - intros Hfree.
    apply andb_true_iff in Hfree as [Hfree Hrest].
    apply andb_true_iff in Hfree as [Hlifting Hknown].
    apply (reaches_trans _ _ _ (with_lifting_observed_reaches L e Hlifting)).
    apply (reaches_trans _
      (normalise_lifted_own (made_closures_called L) (own_closures_known L)
         rounds avoid kept (with_lifting_observed L e))).
    + apply refines_reaches.
      exact (normalise_lifted_own_refines_followed (made_closures_called L)
               (own_closures_known L) rounds avoid kept _ Hknown).
    + exact (IH _ Hrest).
Qed.

Lemma without_statics_refines_followed (D : dropped) (e : expr) :
  refines (State e [] conts statics empty_heap)
    (State (without_statics D e) [] conts statics empty_heap).
Proof.
  unfold without_statics.
  destruct (drop_statics D e) as [e' | ] eqn:Hdrop; [ | apply refines_refl ].
  apply (refines_trans _ _ _
    (drop_statics_refines D e e' Hdrop [] conts statics empty_heap
       (envs_free_of_followed D))).
  apply (expr_equivalent_refines _ _
    (expr_equivalent_trans _ _ _ (drop_deleted_code_equivalent e')
       (merge_statics_equivalent _))).
Qed.

Lemma without_head_statics_refines_followed (D : dropped) (e : expr) :
  head_removal_free_in statics D e = true ->
  refines (State (without_head_statics D e) [] conts statics empty_heap)
    (State e [] conts statics empty_heap).
Proof.
  intros Hhead. unfold without_head_statics.
  destruct (head_statics_removable D e) eqn:Hchecks; [ | apply refines_refl ].
  unfold without_statics.
  destruct (drop_statics D e) as [e' | ] eqn:Hdrop; [ | apply refines_refl ].
  assert (Hdropped : drop_head_statics D e = Some e')
    by (unfold drop_head_statics; rewrite Hchecks; exact Hdrop).
  unfold head_removal_free_in in Hhead. rewrite Hdropped in Hhead.
  apply (refines_trans _ (State e' [] conts statics empty_heap));
    [ | exact (drop_head_statics_refines_under D e e' conts statics
                 empty_heap Hdropped (envs_free_of_followed D) Hhead) ].
  apply (expr_equivalent_refines _ _
    (expr_equivalent_sym _ _
       (expr_equivalent_trans _ _ _ (drop_deleted_code_equivalent e')
          (merge_statics_equivalent _)))).
Qed.

Lemma with_code_first_equivalent_followed (e : expr) :
  equivalent (State e [] conts statics empty_heap)
    (State (with_code_first e) [] conts statics empty_heap).
Proof.
  unfold with_code_first.
  destruct (hoist_code e) as [e' | ] eqn:Hhoist; [ | apply equivalent_refl ].
  apply (equivalent_trans _ _ _
    (hoist_code_equivalent_observed imports conts e e' Hhoist Hobservers)).
  apply (equivalent_trans _ _ _
    (drop_deleted_code_equivalent e' [] conts _ _)).
  apply merge_statics_equivalent.
Qed.

Lemma without_slots_refines_followed (e : expr) :
  refines (State e [] conts statics empty_heap)
    (State (without_slots slots e) [] conts statics empty_heap).
Proof.
  unfold without_slots.
  destruct (drop_unused_slots slots e) as [e' | ] eqn:Hdrop;
    [ | apply refines_refl ].
  apply (refines_trans _ _ _
    (drop_unused_slots_refines_observed imports slots conts e e' Hdrop
       Hfollowed)).
  assert (H : expr_refines e'
                (rewrite_code remove_unused_lets (remove_unused_lets e'))).
  { eapply expr_refines_trans; [ apply remove_unused_lets_refines | ].
    apply rewrite_code_refines, remove_unused_lets_refines. }
  exact (H [] conts statics empty_heap).
Qed.

Lemma with_statics_first_equivalent_followed (Ls : list lifting) (e : expr) :
  statics_first_free_in statics Ls e = true ->
  equivalent (State e [] conts statics empty_heap)
    (State (with_statics_first Ls e) [] conts statics empty_heap).
Proof.
  unfold with_statics_first, statics_first_free_in.
  destruct Ls as [ | L Ls]; [ intros _; apply equivalent_refl | ].
  destruct (hoist_statics (lifting_unit L) e) as [e' | ] eqn:Hhoist;
    [ | intros _; apply equivalent_refl ].
  intros Hfree.
  apply (equivalent_trans _ _ _
    (hoist_statics_equivalent_observed imports conts _ e e' Hhoist Hfree
       Hobservers)).
  apply (equivalent_trans _ _ _
    (drop_deleted_code_equivalent e' [] conts _ _)).
  apply merge_statics_equivalent.
Qed.

(* ------------------------------------------------------------------------ *)
(* The check                                                                *)
(* ------------------------------------------------------------------------ *)

Lemma validate_unit_related_reaches (rounds : nat) (unused : dropped)
    (G : loopified) (Ls : list lifting) (kept : list code_id)
    (before after : expr) :
  validate_unit_related imports rounds unused slots G Ls kept before after
  = Related_once_lifted ->
  reaches before after.
Proof.
  unfold validate_unit_related. cbv zeta. fold statics.
  set (source := loopify_unit G (cut_returns [] (demote_stub_calls before))).
  set (a := tidied_after rounds (normalise_after rounds kept after)).
  set (taken := negb (keeps_a_return a)).
  set (c := without_head_statics unused a).
  set (lifted :=
    without_statics unused
      (without_slots slots
         (tidied_before rounds
            (with_returns taken
               (lifted_in_turn_observed rounds (unit_continuations Ls) Ls
                  kept source))))).
  destruct
    (code_closed before && code_closed after
     && head_removal_free_in statics unused a
     && lifted_observed_free_in statics rounds (unit_continuations Ls) Ls
          kept source
     && (expr_eqb [] []
           (order_statics_as (float_code (with_code_first c))
              (float_code lifted))
           (float_code (with_code_first c))
         || statics_first_free_in statics Ls c
            && statics_first_free_in statics Ls lifted
            && expr_eqb [] []
                 (order_statics_as (with_statics_first Ls c)
                    (with_statics_first Ls lifted))
                 (with_statics_first Ls c))) eqn:Htests;
    [ intros _ | discriminate ].
  apply andb_true_iff in Htests as [Htests Hsame].
  apply andb_true_iff in Htests as [Htests Hlifted_free].
  apply andb_true_iff in Htests as [_ Hhead].
  (* the unit before, to the lifted term *)
  apply (reaches_trans _ source).
  { apply refines_reaches.
    apply (refines_trans _ _ _
      (demote_stub_calls_refines before [] conts statics empty_heap)).
    apply (refines_trans _ _ _
      (cut_returns_refines (demote_stub_calls before) [] conts statics
         empty_heap)).
    exact (loopify_unit_refines G (cut_returns [] (demote_stub_calls before))
             [] conts statics empty_heap). }
  apply (reaches_trans _ _ _
    (lifted_in_turn_observed_reaches rounds (unit_continuations Ls) Ls kept
       source Hlifted_free)).
  apply refines_reaches.
  apply (refines_trans _ (State lifted [] conts statics empty_heap)).
  { apply (refines_trans _ _ _
      (with_returns_refines taken _ [] conts statics empty_heap)).
    apply (refines_trans _ _ _
      (tidied_before_refines rounds _ [] conts statics empty_heap)).
    apply (refines_trans _ _ _ (without_slots_refines_followed _)).
    exact (without_statics_refines_followed unused _). }
  (* the unit after, from the term compared *)
  assert (Hcleaned :
    refines (State c [] conts statics empty_heap)
      (State after [] conts statics empty_heap)).
  { apply (refines_trans _ _ _
      (without_head_statics_refines_followed unused a Hhead)).
    apply equivalent_refines, equivalent_sym.
    exact (tidied_after_normalised_equivalent rounds kept after [] conts
             statics empty_heap). }
  apply orb_true_iff in Hsame as [Hsame | Hsame].
  - (* the first comparison: the code of the unit after first *)
    apply (refines_trans _
      (State (with_code_first c) [] conts statics empty_heap)).
    { apply expr_equivalent_refines.
      apply (expr_equivalent_trans _ _ _ (float_code_equivalent _)).
      apply (expr_equivalent_trans _ _ _
        (order_statics_as_equivalent (float_code (with_code_first c)) _)).
      apply (expr_equivalent_trans _ _ _ (expr_eqb_sound _ _ Hsame)).
      apply expr_equivalent_sym, float_code_equivalent. }
    apply (refines_trans _ (State c [] conts statics empty_heap));
      [ | exact Hcleaned ].
    apply equivalent_refines, equivalent_sym.
    apply with_code_first_equivalent_followed.
  - (* the second: the static bindings that need nothing first *)
    apply andb_true_iff in Hsame as [Hfree Hsame].
    apply andb_true_iff in Hfree as [Hfree_after Hfree_lifted].
    apply (refines_trans _ _ _
      (equivalent_refines _ _
         (with_statics_first_equivalent_followed Ls lifted Hfree_lifted))).
    apply (refines_trans _
      (State (with_statics_first Ls c) [] conts statics empty_heap)).
    { apply expr_equivalent_refines.
      apply (expr_equivalent_trans _ _ _
        (order_statics_as_equivalent (with_statics_first Ls c) _)).
      exact (expr_eqb_sound _ _ Hsame). }
    apply (refines_trans _ (State c [] conts statics empty_heap));
      [ | exact Hcleaned ].
    apply equivalent_refines, equivalent_sym.
    exact (with_statics_first_equivalent_followed Ls c Hfree_after).
Qed.

End Followed.

Theorem validate_unit_related_under_observers :
  validate_unit_related_under_observers_statement.
Proof.
  intros imports rounds unused slots G Ls kept before after Hrelated conts
    Hobservers.
  exact (validate_unit_related_reaches imports slots conts Hobservers rounds
           unused G Ls kept before after Hrelated).
Qed.

Lemma validate_unit_related_code_closed
    (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr) :
  validate_unit_related imports rounds unused slots G Ls kept before after
  = Related_once_lifted ->
  code_closed before = true ∧ code_closed after = true.
Proof.
  unfold validate_unit_related. cbv zeta.
  destruct (code_closed before); [ | discriminate ].
  destruct (code_closed after); [ | discriminate ].
  intros _. split; reflexivity.
Qed.

Theorem validate_unit_related_sound : validate_unit_related_sound_statement.
Proof.
  intros imports rounds unused slots G Ls kept before after Hrelated.
  destruct (validate_unit_related_code_closed imports rounds unused slots G
              Ls kept before after Hrelated) as [Hbefore Hafter].
  apply (proj1 (under_observers_worlds slots (imports_statics imports)
                  empty_heap before after Hbefore Hafter
                  (Forall_nil _))).
  exact (validate_unit_related_under_observers imports rounds unused slots G
           Ls kept before after Hrelated).
Qed.
