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

(* Proofs of the statements of FlambdaRenameApartAlphaSpec.v. Nothing
   here needs review.

   The recursion is on the term, and what is proved is of the term
   under the renaming that [apart] has made so far: the term renamed
   by it is alpha-equivalent to what [apart] gives. The new names are
   distinct, outside every name of the renamed term, and at least as
   many as the term has binders; a binder takes those at the front and
   leaves the rest with that property for what follows.

   At a binder the renaming so far, [r], becomes [r'], which sends the
   bound names to the new names and is [r] elsewhere
   ([bind_variables_bound], [bind_variables_unbound]). The rule of
   [alpha_equivalent] for that binder is used at the renaming that
   undoes [r] and then does [r'] ([moved]): it fixes the free names of
   the scope, since they are not new names, and it turns the scope
   renamed by [r] into the scope renamed by [r']. The recursion then
   gives the scope under [r'], inside the binder with its new names
   ([Alpha_context]). *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaBindersSpec FlambdaCompareSpec
  FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaSpec
  FlambdaAlphaProofs FlambdaContextSpec FlambdaCompareAlphaSpec
  FlambdaCompareAlphaProofs FlambdaNameAbstractionSpec
  FlambdaFreshQuantifierSpec FlambdaFreshQuantifierProofs
  FlambdaFreshBindersSpec FlambdaRenameApartSpec FlambdaRenameApartProofs
  FlambdaRenameApartAlphaSpec FlambdaTermInductionProofs.
Import ListNotations.

(* ======================================================================== *)
(* Swaps and lists                                                          *)
(* ======================================================================== *)

Lemma swap_cases (a b c : positive) :
  swap a b c = b ∨ swap a b c = a ∨ swap a b c = c.
Proof.
  unfold swap. destruct (Pos.eqb c a); [ left; reflexivity | ].
  destruct (Pos.eqb c b); [ right; left | right; right ]; reflexivity.
Qed.

Lemma swap_other (a b c : positive) : c <> a -> c <> b -> swap a b c = c.
Proof.
  intros Ha Hb. unfold swap.
  destruct (Pos.eqb_spec c a); [ contradiction | ].
  destruct (Pos.eqb_spec c b); [ contradiction | reflexivity ].
Qed.

Lemma swap_first (a b : positive) : swap a b a = b.
Proof. unfold swap. rewrite Pos.eqb_refl. reflexivity. Qed.

(* The front and the rest of a list that repeats nothing share
   nothing. *)
Lemma NoDup_front_rest {A : Type} (n : nat) (l : list A) (a : A) :
  NoDup l -> In a (firstn n l) -> In a (skipn n l) -> False.
Proof.
  intros Hnodup Hfront Hrest.
  rewrite <- (firstn_skipn n l) in Hnodup.
  revert Hnodup Hfront. generalize (firstn n l) as front.
  induction front as [ | b front IH]; simpl; intros Hnodup Hfront;
    [ contradiction | ].
  inversion Hnodup as [ | b' l' Hb Hnodup']; subst.
  destruct Hfront as [-> | Hfront]; [ | exact (IH Hnodup' Hfront) ].
  apply Hb, in_or_app. right. exact Hrest.
Qed.

Lemma NoDup_firstn {A : Type} (n : nat) (l : list A) :
  NoDup l -> NoDup (firstn n l).
Proof.
  intros Hnodup. rewrite <- (firstn_skipn n l) in Hnodup.
  exact (NoDup_app_remove_r _ _ Hnodup).
Qed.

Lemma NoDup_skipn {A : Type} (n : nat) (l : list A) :
  NoDup l -> NoDup (skipn n l).
Proof.
  intros Hnodup. rewrite <- (firstn_skipn n l) in Hnodup.
  exact (NoDup_app_remove_l _ _ Hnodup).
Qed.

Lemma in_firstn {A : Type} (a : A) (n : nat) (l : list A) :
  In a (firstn n l) -> In a l.
Proof.
  intros Ha. rewrite <- (firstn_skipn n l). apply in_or_app. left. exact Ha.
Qed.

(* ======================================================================== *)
(* The renaming with the names of a binder bound                            *)
(* ======================================================================== *)

(* The new names taken are those at the front. *)
Lemma bind_variables_taken (xs : list variable) :
  ∀ (r : renaming) (news : list positive),
    length xs <= length news ->
    snd (fst (bind_variables r news xs)) = firstn (length xs) news.
Proof.
  induction xs as [ | x xs IH]; intros r news Hlength;
    cbn [bind_variables length]; [ reflexivity | ].
  destruct news as [ | a news]; [ simpl in Hlength; lia | ].
  cbn [new_variable hd tl firstn].
  pose proof (IH (bind_variable r x a) news) as Htaken.
  destruct (bind_variables (bind_variable r x a) news xs) as [[rest xs'] r'].
  cbn [fst snd] in Htaken |- *. rewrite Htaken; [ reflexivity | ].
  simpl in Hlength. lia.
Qed.

(* The continuations are renamed as they were. *)
Lemma bind_variables_continuations (xs : list variable) :
  ∀ (r : renaming) (news : list positive),
    continuation_permutation (snd (bind_variables r news xs))
    = continuation_permutation r.
Proof.
  induction xs as [ | x xs IH]; intros r news; cbn [bind_variables];
    [ reflexivity | ].
  pose proof (IH (bind_variable r x (new_variable r news x)) (tl news))
    as Hsame.
  destruct
    (bind_variables (bind_variable r x (new_variable r news x)) (tl news) xs)
    as [[rest xs'] r'].
  cbn [snd] in Hsame |- *. exact Hsame.
Qed.

(* Where the renaming sends a variable: to a new name taken, or where
   the renaming before sent it or one of the variables bound. *)
Lemma bind_variables_range (xs : list variable) :
  ∀ (r : renaming) (news : list positive) (y : variable),
    length xs <= length news ->
    In (rename_variable (snd (bind_variables r news xs)) y)
      (firstn (length xs) news ++ map (rename_variable r) (y :: xs)).
Proof.
  induction xs as [ | x xs IH]; intros r news y Hlength;
    cbn [bind_variables length]; [ left; reflexivity | ].
  destruct news as [ | a news]; [ simpl in Hlength; lia | ].
  cbn [new_variable hd tl firstn].
  assert (Hlength' : length xs <= length news) by (simpl in Hlength; lia).
  pose proof (IH (bind_variable r x a) news y Hlength') as Hrange.
  destruct (bind_variables (bind_variable r x a) news xs) as [[rest xs'] r'].
  cbn [snd] in Hrange |- *.
  apply in_app_or in Hrange. destruct Hrange as [Hrange | Hrange];
    [ right; apply in_or_app; left; exact Hrange | ].
  apply in_map_iff in Hrange. destruct Hrange as (z & Hz & Hin).
  rewrite rename_variable_bind_variable in Hz.
  destruct (swap_cases (rename_variable r x) a (rename_variable r z))
    as [Hswap | [Hswap | Hswap]];
    rewrite Hswap in Hz; rewrite <- Hz.
  - left. reflexivity.
  - right. apply in_or_app. right. right. left. reflexivity.
  - right. apply in_or_app. right.
    destruct Hin as [-> | Hin]; [ left; reflexivity | ].
    right. right. apply in_map. exact Hin.
Qed.

(* A variable that is not bound, and that the renaming before does not
   send to a new name taken, is renamed as it was. *)
Lemma bind_variables_unbound (xs : list variable) :
  ∀ (r : renaming) (news : list positive) (y : variable),
    length xs <= length news ->
    ~ In y xs ->
    ~ In (rename_variable r y) (firstn (length xs) news) ->
    rename_variable (snd (bind_variables r news xs)) y = rename_variable r y.
Proof.
  induction xs as [ | x xs IH]; intros r news y Hlength Hy Hnew;
    cbn [bind_variables length]; [ reflexivity | ].
  destruct news as [ | a news]; [ simpl in Hlength; lia | ].
  cbn [new_variable hd tl firstn length] in Hnew |- *.
  assert (Hlength' : length xs <= length news) by (simpl in Hlength; lia).
  assert (Hstep :
    rename_variable (bind_variable r x a) y = rename_variable r y).
  { rewrite rename_variable_bind_variable. apply swap_other.
    - intros Heq. apply Hy. left.
      exact (rename_variable_injective r _ _ (eq_sym Heq)).
    - intros Heq. apply Hnew. left. symmetry. exact Heq. }
  pose proof (IH (bind_variable r x a) news y Hlength') as Hrest.
  destruct (bind_variables (bind_variable r x a) news xs) as [[rest xs'] r'].
  cbn [snd] in Hrest |- *. rewrite <- Hstep. apply Hrest.
  - intros Hin. apply Hy. right. exact Hin.
  - rewrite Hstep. intros Hin. apply Hnew. right. exact Hin.
Qed.

(* The variables bound, which are distinct, are sent to the new names
   taken, which are distinct and none of which the renaming before
   sends a bound variable to. *)
Lemma bind_variables_bound (xs : list variable) :
  ∀ (r : renaming) (news : list positive),
    length xs <= length news ->
    NoDup xs ->
    NoDup (firstn (length xs) news) ->
    (∀ a : positive,
      In a (firstn (length xs) news) ->
      ~ In a (map (rename_variable r) xs)) ->
    map (rename_variable (snd (bind_variables r news xs))) xs
    = firstn (length xs) news.
Proof.
  induction xs as [ | x xs IH]; intros r news Hlength Hxs Hnews Houtside;
    [ reflexivity | ].
  destruct news as [ | a news]; [ simpl in Hlength; lia | ].
  assert (Hlength' : length xs <= length news) by (simpl in Hlength; lia).
  cbn [length firstn] in Hnews, Houtside.
  inversion Hxs as [ | x' xs' Hx Hxs']; subst.
  inversion Hnews as [ | a' news' Ha Hnews']; subst.
  assert (Hfirst : rename_variable (bind_variable r x a) x = a).
  { rewrite rename_variable_bind_variable. apply swap_first. }
  pose proof (bind_variables_unbound xs (bind_variable r x a) news x Hlength')
    as Hunbound.
  pose proof (IH (bind_variable r x a) news Hlength' Hxs' Hnews') as Hrest.
  cbn [bind_variables length new_variable hd tl firstn map].
  destruct (bind_variables (bind_variable r x a) news xs) as [[rest xs'] r'].
  cbn [snd] in Hunbound, Hrest |- *.
  rewrite Hunbound, Hfirst;
    [ | exact Hx | rewrite Hfirst; exact Ha ].
  f_equal. apply Hrest. intros b Hb Hin.
  apply in_map_iff in Hin. destruct Hin as (z & Hz & Hin).
  rewrite rename_variable_bind_variable, swap_other in Hz.
  - apply (Houtside b (or_intror Hb)). right. rewrite <- Hz.
    apply in_map. exact Hin.
  - intros Heq. apply rename_variable_injective in Heq. subst z.
    exact (Hx Hin).
  - intros Heq. apply (Houtside a (or_introl eq_refl)). right.
    rewrite <- Heq. apply in_map. exact Hin.
Qed.

(* The same for the continuations of a binder. *)
Lemma bind_continuations_taken (ks : list continuation) :
  ∀ (r : renaming) (news : list positive),
    length ks <= length news ->
    snd (fst (bind_continuations r news ks)) = firstn (length ks) news.
Proof.
  induction ks as [ | k ks IH]; intros r news Hlength;
    cbn [bind_continuations length]; [ reflexivity | ].
  destruct news as [ | a news]; [ simpl in Hlength; lia | ].
  cbn [new_continuation hd tl firstn].
  pose proof (IH (bind_continuation r k a) news) as Htaken.
  destruct (bind_continuations (bind_continuation r k a) news ks)
    as [[rest ks'] r'].
  cbn [fst snd] in Htaken |- *. rewrite Htaken; [ reflexivity | ].
  simpl in Hlength. lia.
Qed.

(* The variables are renamed as they were. *)
Lemma bind_continuations_variables (ks : list continuation) :
  ∀ (r : renaming) (news : list positive),
    variable_permutation (snd (bind_continuations r news ks))
    = variable_permutation r.
Proof.
  induction ks as [ | k ks IH]; intros r news; cbn [bind_continuations];
    [ reflexivity | ].
  pose proof (IH (bind_continuation r k (new_continuation r news k)) (tl news))
    as Hsame.
  destruct
    (bind_continuations (bind_continuation r k (new_continuation r news k))
       (tl news) ks)
    as [[rest ks'] r'].
  cbn [snd] in Hsame |- *. exact Hsame.
Qed.

(* Where the renaming sends a continuation, likewise. *)
Lemma bind_continuations_range (ks : list continuation) :
  ∀ (r : renaming) (news : list positive) (j : continuation),
    length ks <= length news ->
    In (rename_continuation (snd (bind_continuations r news ks)) j)
      (firstn (length ks) news ++ map (rename_continuation r) (j :: ks)).
Proof.
  induction ks as [ | k ks IH]; intros r news j Hlength;
    cbn [bind_continuations length]; [ left; reflexivity | ].
  destruct news as [ | a news]; [ simpl in Hlength; lia | ].
  cbn [new_continuation hd tl firstn].
  assert (Hlength' : length ks <= length news) by (simpl in Hlength; lia).
  pose proof (IH (bind_continuation r k a) news j Hlength') as Hrange.
  destruct (bind_continuations (bind_continuation r k a) news ks)
    as [[rest ks'] r'].
  cbn [snd] in Hrange |- *.
  apply in_app_or in Hrange. destruct Hrange as [Hrange | Hrange];
    [ right; apply in_or_app; left; exact Hrange | ].
  apply in_map_iff in Hrange. destruct Hrange as (z & Hz & Hin).
  rewrite rename_continuation_bind_continuation in Hz.
  destruct (swap_cases (rename_continuation r k) a (rename_continuation r z))
    as [Hswap | [Hswap | Hswap]];
    rewrite Hswap in Hz; rewrite <- Hz.
  - left. reflexivity.
  - right. apply in_or_app. right. right. left. reflexivity.
  - right. apply in_or_app. right.
    destruct Hin as [-> | Hin]; [ left; reflexivity | ].
    right. right. apply in_map. exact Hin.
Qed.

(* A continuation that is not bound, likewise. *)
Lemma bind_continuations_unbound (ks : list continuation) :
  ∀ (r : renaming) (news : list positive) (j : continuation),
    length ks <= length news ->
    ~ In j ks ->
    ~ In (rename_continuation r j) (firstn (length ks) news) ->
    rename_continuation (snd (bind_continuations r news ks)) j
    = rename_continuation r j.
Proof.
  induction ks as [ | k ks IH]; intros r news j Hlength Hj Hnew;
    cbn [bind_continuations length]; [ reflexivity | ].
  destruct news as [ | a news]; [ simpl in Hlength; lia | ].
  cbn [new_continuation hd tl firstn length] in Hnew |- *.
  assert (Hlength' : length ks <= length news) by (simpl in Hlength; lia).
  assert (Hstep :
    rename_continuation (bind_continuation r k a) j = rename_continuation r j).
  { rewrite rename_continuation_bind_continuation. apply swap_other.
    - intros Heq. apply Hj. left.
      exact (rename_continuation_injective r _ _ (eq_sym Heq)).
    - intros Heq. apply Hnew. left. symmetry. exact Heq. }
  pose proof (IH (bind_continuation r k a) news j Hlength') as Hrest.
  destruct (bind_continuations (bind_continuation r k a) news ks)
    as [[rest ks'] r'].
  cbn [snd] in Hrest |- *. rewrite <- Hstep. apply Hrest.
  - intros Hin. apply Hj. right. exact Hin.
  - rewrite Hstep. intros Hin. apply Hnew. right. exact Hin.
Qed.

(* The continuations bound, likewise. *)
Lemma bind_continuations_bound (ks : list continuation) :
  ∀ (r : renaming) (news : list positive),
    length ks <= length news ->
    NoDup ks ->
    NoDup (firstn (length ks) news) ->
    (∀ a : positive,
      In a (firstn (length ks) news) ->
      ~ In a (map (rename_continuation r) ks)) ->
    map (rename_continuation (snd (bind_continuations r news ks))) ks
    = firstn (length ks) news.
Proof.
  induction ks as [ | k ks IH]; intros r news Hlength Hks Hnews Houtside;
    [ reflexivity | ].
  destruct news as [ | a news]; [ simpl in Hlength; lia | ].
  assert (Hlength' : length ks <= length news) by (simpl in Hlength; lia).
  cbn [length firstn] in Hnews, Houtside.
  inversion Hks as [ | k' ks' Hk Hks']; subst.
  inversion Hnews as [ | a' news' Ha Hnews']; subst.
  assert (Hfirst : rename_continuation (bind_continuation r k a) k = a).
  { rewrite rename_continuation_bind_continuation. apply swap_first. }
  pose proof
    (bind_continuations_unbound ks (bind_continuation r k a) news k Hlength')
    as Hunbound.
  pose proof (IH (bind_continuation r k a) news Hlength' Hks' Hnews') as Hrest.
  cbn [bind_continuations length new_continuation hd tl firstn map].
  destruct (bind_continuations (bind_continuation r k a) news ks)
    as [[rest ks'] r'].
  cbn [snd] in Hunbound, Hrest |- *.
  rewrite Hunbound, Hfirst;
    [ | exact Hk | rewrite Hfirst; exact Ha ].
  f_equal. apply Hrest. intros b Hb Hin.
  apply in_map_iff in Hin. destruct Hin as (z & Hz & Hin).
  rewrite rename_continuation_bind_continuation, swap_other in Hz.
  - apply (Houtside b (or_intror Hb)). right. rewrite <- Hz.
    apply in_map. exact Hin.
  - intros Heq. apply rename_continuation_injective in Heq. subst z.
    exact (Hk Hin).
  - intros Heq. apply (Houtside a (or_introl eq_refl)). right.
    rewrite <- Heq. apply in_map. exact Hin.
Qed.

(* ======================================================================== *)
(* From one renaming to the next                                            *)
(* ======================================================================== *)

(* The renaming that undoes [r] and then does [r']. *)
Definition moved (r r' : renaming) : renaming :=
  compose_renaming r' (inverse_renaming r).

Lemma moved_variable (r r' : renaming) (x : variable) :
  rename_variable (moved r r') (rename_variable r x) = rename_variable r' x.
Proof.
  unfold moved. rewrite rename_variable_compose, rename_variable_inverse_l.
  reflexivity.
Qed.

Lemma moved_continuation (r r' : renaming) (k : continuation) :
  rename_continuation (moved r r') (rename_continuation r k)
  = rename_continuation r' k.
Proof.
  unfold moved.
  rewrite rename_continuation_compose, rename_continuation_inverse_l.
  reflexivity.
Qed.

Lemma moved_expr (r r' : renaming) (e : expr) :
  rename_expr (moved r r') (rename_expr r e) = rename_expr r' e.
Proof.
  rewrite <- rename_expr_compose_aux. apply rename_expr_ext.
  - intros x. rewrite rename_variable_compose. apply moved_variable.
  - intros k. rewrite rename_continuation_compose. apply moved_continuation.
Qed.

Lemma moved_handler (r r' : renaming) (h : continuation_handler) :
  rename_handler (moved r r') (rename_handler r h) = rename_handler r' h.
Proof.
  rewrite <- rename_handler_compose. apply rename_handler_ext.
  - intros x. rewrite rename_variable_compose. apply moved_variable.
  - intros k. rewrite rename_continuation_compose. apply moved_continuation.
Qed.

Lemma moved_code (r r' : renaming) (c : code) :
  rename_code (moved r r') (rename_code r c) = rename_code r' c.
Proof.
  rewrite <- rename_code_compose. apply rename_code_ext.
  - intros x. rewrite rename_variable_compose. apply moved_variable.
  - intros k. rewrite rename_continuation_compose. apply moved_continuation.
Qed.

(* It fixes a name that the two renamings send to the same name. *)
Lemma moved_fixes_variable (r r' : renaming) (x : variable) :
  rename_variable r' x = rename_variable r x ->
  rename_variable (moved r r') (rename_variable r x) = rename_variable r x.
Proof. intros Hsame. rewrite moved_variable. exact Hsame. Qed.

Lemma moved_fixes_continuation (r r' : renaming) (k : continuation) :
  rename_continuation r' k = rename_continuation r k ->
  rename_continuation (moved r r') (rename_continuation r k)
  = rename_continuation r k.
Proof. intros Hsame. rewrite moved_continuation. exact Hsame. Qed.

(* Every name is what a renaming sends some name to. *)
Lemma variable_renamed (r : renaming) (y : variable) :
  y = rename_variable r (rename_variable (inverse_renaming r) y).
Proof. symmetry. apply rename_variable_inverse_r. Qed.

Lemma continuation_renamed (r : renaming) (k : continuation) :
  k = rename_continuation r (rename_continuation (inverse_renaming r) k).
Proof. symmetry. apply rename_continuation_inverse_r. Qed.

Lemma swap_variable_first (x a : variable) :
  rename_variable (swap_variable x a) x = a.
Proof. unfold rename_variable, swap_variable. simpl. apply swap_first. Qed.

Lemma swap_continuation_first (k a : continuation) :
  rename_continuation (swap_continuation k a) k = a.
Proof.
  unfold rename_continuation, swap_continuation. simpl. apply swap_first.
Qed.

Lemma among_without_in (name : positive) (bound names : list positive) :
  among name (without bound names) = true ->
  In name names ∧ ~ In name bound.
Proof.
  rewrite among_without. intros Hname.
  apply andb_true_iff in Hname as [Hnames Hbound].
  split; [ apply among_In, Hnames | ].
  intros Hin. apply among_In in Hin. rewrite Hin in Hbound.
  discriminate Hbound.
Qed.

(* ======================================================================== *)
(* The new names                                                            *)
(* ======================================================================== *)

(* The new names are distinct, none of them is one of the variables or
   of the continuations given, and there are [count] of them at
   least. *)
Definition ready (news : list positive) (count : nat) (vs : list variable)
    (ks : list continuation) : Prop :=
  NoDup news
  ∧ (∀ a : positive, In a news -> ~ In a vs ∧ ~ In a ks)
  ∧ count <= length news.

(* Once variables are bound, the new names left are outside what the
   new renaming makes of the names of the scope. *)
Lemma ready_after_variables (xs : list variable) (r : renaming)
    (news : list positive) (count : nat) (vs : list variable)
    (ks : list continuation) :
  NoDup news ->
  length xs + count <= length news ->
  (∀ a : positive,
    In a news ->
    ~ In a (map (rename_variable r) (xs ++ vs))
    ∧ ~ In a (map (rename_continuation r) ks)) ->
  ready (skipn (length xs) news) count
    (map (rename_variable (snd (bind_variables r news xs))) vs)
    (map (rename_continuation (snd (bind_variables r news xs))) ks).
Proof.
  intros Hnodup Hcount Houtside.
  split; [ apply NoDup_skipn, Hnodup | ].
  split; [ | rewrite length_skipn; lia ].
  intros b Hb.
  assert (Hin : In b news) by exact (in_skipn b (length xs) news Hb).
  split.
  - intros Hmap. apply in_map_iff in Hmap. destruct Hmap as (y & Hy & Hvs).
    pose proof (bind_variables_range xs r news y ltac:(lia)) as Hrange.
    rewrite Hy in Hrange. apply in_app_or in Hrange.
    destruct Hrange as [Hfront | Hold];
      [ exact (NoDup_front_rest (length xs) news b Hnodup Hfront Hb) | ].
    apply (proj1 (Houtside b Hin)). rewrite map_app. apply in_or_app.
    destruct Hold as [Hold | Hold];
      [ right; rewrite <- Hold; apply in_map; exact Hvs | left; exact Hold ].
  - unfold rename_continuation. rewrite bind_variables_continuations.
    exact (proj2 (Houtside b Hin)).
Qed.

Lemma ready_after_continuations (js : list continuation) (r : renaming)
    (news : list positive) (count : nat) (vs : list variable)
    (ks : list continuation) :
  NoDup news ->
  length js + count <= length news ->
  (∀ a : positive,
    In a news ->
    ~ In a (map (rename_variable r) vs)
    ∧ ~ In a (map (rename_continuation r) (js ++ ks))) ->
  ready (skipn (length js) news) count
    (map (rename_variable (snd (bind_continuations r news js))) vs)
    (map (rename_continuation (snd (bind_continuations r news js))) ks).
Proof.
  intros Hnodup Hcount Houtside.
  split; [ apply NoDup_skipn, Hnodup | ].
  split; [ | rewrite length_skipn; lia ].
  intros b Hb.
  assert (Hin : In b news) by exact (in_skipn b (length js) news Hb).
  split.
  - unfold rename_variable. rewrite bind_continuations_variables.
    exact (proj1 (Houtside b Hin)).
  - intros Hmap. apply in_map_iff in Hmap. destruct Hmap as (j & Hj & Hks).
    pose proof (bind_continuations_range js r news j ltac:(lia)) as Hrange.
    rewrite Hj in Hrange. apply in_app_or in Hrange.
    destruct Hrange as [Hfront | Hold];
      [ exact (NoDup_front_rest (length js) news b Hnodup Hfront Hb) | ].
    apply (proj2 (Houtside b Hin)). rewrite map_app. apply in_or_app.
    destruct Hold as [Hold | Hold];
      [ right; rewrite <- Hold; apply in_map; exact Hks | left; exact Hold ].
Qed.

(* ======================================================================== *)
(* What the recursion proves                                                *)
(* ======================================================================== *)

(* Of an expression, under the renaming so far. *)
Definition apart_alpha (e : expr) : Prop :=
  ∀ (r : renaming) (news : list positive),
    code_closed e = true ->
    binders_distinct e = true ->
    ready news (binders e) (variables_of (rename_expr r e))
      (continuations_of (rename_expr r e)) ->
    alpha_equivalent (rename_expr r e) (snd (apart r news e)).

(* A handler may be replaced by another wherever it stands, in a
   non-recursive [Let_cont] or in a recursive one. *)
Definition handler_replaceable (h h' : continuation_handler) : Prop :=
  (∀ (k : continuation) (body : expr),
    alpha_equivalent (Let_cont_non_recursive k h body)
      (Let_cont_non_recursive k h' body))
  ∧ (∀ (before : list (continuation * continuation_handler))
       (k : continuation)
       (after : list (continuation * continuation_handler)) (body : expr),
      alpha_equivalent
        (Let_cont_recursive (before ++ (k, h) :: after) body)
        (Let_cont_recursive (before ++ (k, h') :: after) body)).

Definition apart_handler_alpha (h : continuation_handler) : Prop :=
  ∀ (r : renaming) (news : list positive),
    handler_code_closed h = true ->
    handler_binders_distinct h = true ->
    ready news (binders_in_handler h)
      (variables_in_handler (rename_handler r h))
      (continuations_in_handler (rename_handler r h)) ->
    handler_replaceable (rename_handler r h) (snd (apart_handler r news h)).

(* A static binding may be replaced by another wherever it stands. *)
Definition binding_replaceable (binding binding' : static_binding) : Prop :=
  ∀ (before after : list static_binding) (body : expr),
    alpha_equivalent (Let_static (before ++ binding :: after) body)
      (Let_static (before ++ binding' :: after) body).

Definition apart_binding_alpha (binding : static_binding) : Prop :=
  ∀ (r : renaming) (news : list positive),
    binding_code_closed binding = true ->
    binding_binders_distinct binding = true ->
    ready news (binders_in_binding binding)
      (variables_in_binding (rename_static_binding r binding))
      (continuations_in_binding (rename_static_binding r binding)) ->
    binding_replaceable (rename_static_binding r binding)
      (snd (apart_binding r news binding)).

Definition apart_code_alpha (c : code) : Prop :=
  ∀ (r : renaming) (news : list positive) (id : code_id),
    code_is_closed c = true ->
    code_binders_distinct c = true ->
    ready news (binders_in_code c) (variables_in_code (rename_code r c))
      (continuations_in_code (rename_code r c)) ->
    binding_replaceable (Bind_code id (rename_code r c))
      (Bind_code id (snd (apart_code r news c))).

(* ======================================================================== *)
(* What has no binder                                                       *)
(* ======================================================================== *)

Lemma apply_apart_alpha (ap : apply_expr) : apart_alpha (Apply ap).
Proof. intros r news _ _ _. apply Alpha_refl. Qed.

Lemma apply_cont_apart_alpha (ac : apply_cont) : apart_alpha (Apply_cont ac).
Proof. intros r news _ _ _. apply Alpha_refl. Qed.

Lemma switch_apart_alpha (scrutinee : simple) (arms : list (Z * apply_cont)) :
  apart_alpha (Switch scrutinee arms).
Proof. intros r news _ _ _. apply Alpha_refl. Qed.

Lemma invalid_apart_alpha : apart_alpha Invalid.
Proof. intros r news _ _ _. apply Alpha_refl. Qed.

(* ======================================================================== *)
(* A [Let]                                                                  *)
(* ======================================================================== *)

Lemma let_apart_alpha (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  apart_alpha body -> apart_alpha (Let x kind d body).
Proof.
  intros IH r news Hclosed Hdistinct (Hnodup & Houtside & Hcount).
  cbn [code_closed binders_distinct binders] in Hclosed, Hdistinct, Hcount.
  destruct news as [ | a news]; [ simpl in Hcount; lia | ].
  cbn [rename_expr variables_of continuations_of] in Houtside.
  assert (Ha : In a (a :: news)) by (left; reflexivity).
  (* The new names left, for the body. *)
  assert (Hready :
    ready news (binders body)
      (variables_of (rename_expr (bind_variable r x a) body))
      (continuations_of (rename_expr (bind_variable r x a) body))).
  { rewrite variables_of_renamed, continuations_of_renamed.
    refine
      (ready_after_variables [x] r (a :: news) (binders body) _ _ Hnodup _ _);
      [ simpl in Hcount |- *; lia | ].
    intros b Hb. destruct (Houtside b Hb) as [Hv Hk]. split.
    - intros Hin. apply Hv. cbn [app map] in Hin.
      destruct Hin as [Hin | Hin]; [ left; exact Hin | ].
      right. apply in_or_app. right. rewrite variables_of_renamed.
      exact Hin.
    - rewrite <- continuations_of_renamed. exact Hk. }
  specialize (IH (bind_variable r x a) news Hclosed Hdistinct Hready).
  cbn [rename_expr apart new_variable hd tl].
  destruct (apart (bind_variable r x a) news body) as [rest body'].
  cbn [snd] in IH |- *.
  apply Alpha_trans
    with (Let a kind (rename_named r d)
            (rename_expr (bind_variable r x a) body)).
  - pose proof
      (Alpha_let (swap_variable (rename_variable r x) a)
         (rename_variable r x) kind (rename_named r d) (rename_expr r body))
      as Hstep.
    rewrite swap_variable_first, <- rename_expr_compose_aux in Hstep.
    apply Hstep; [ rewrite code_closed_renamed; exact Hclosed | ].
    split; [ | intros k _; reflexivity ].
    intros y Hy. apply among_without_in in Hy as [Hfree Hbound].
    apply swap_other.
    + intros Heq. apply Hbound. left. symmetry. exact Heq.
    + intros Heq. apply (proj1 (Houtside a Ha)). right. apply in_or_app.
      right. apply free_variables_in_variables_of. rewrite <- Heq.
      exact Hfree.
  - exact (Alpha_context (Let_body a kind (rename_named r d) Hole) _ _ IH).
Qed.

(* ======================================================================== *)
(* A handler                                                                *)
(* ======================================================================== *)

Lemma handler_apart_alpha (params : list variable)
    (metadata : handler_metadata) (handler : expr) :
  apart_alpha handler -> apart_handler_alpha (Handler params metadata handler).
Proof.
  intros IH r news Hclosed Hdistinct (Hnodup & Houtside & Hcount).
  cbn [handler_code_closed handler_binders_distinct binders_in_handler]
    in Hclosed, Hdistinct, Hcount.
  apply andb_true_iff in Hdistinct as [Hparams Hdistinct].
  apply distinct_NoDup in Hparams.
  cbn [rename_handler variables_in_handler continuations_in_handler]
    in Houtside.
  assert (Hlength : length params <= length news) by lia.
  (* What the renaming with the parameters bound does. *)
  pose proof (bind_variables_rest params r news) as Hrest.
  pose proof (bind_variables_taken params r news Hlength) as Htaken.
  pose proof (bind_variables_unbound params r news) as Hunbound.
  pose proof (bind_variables_continuations params r news) as Hconts.
  assert (Hbound :
    map (rename_variable (snd (bind_variables r news params))) params
    = firstn (length params) news).
  { apply (bind_variables_bound params r news Hlength Hparams);
      [ apply NoDup_firstn, Hnodup | ].
    intros a Ha Hin.
    apply (proj1 (Houtside a (in_firstn a _ news Ha))), in_or_app.
    left. exact Hin. }
  assert (Hready :
    ready (skipn (length params) news) (binders handler)
      (map (rename_variable (snd (bind_variables r news params)))
         (variables_of handler))
      (map (rename_continuation (snd (bind_variables r news params)))
         (continuations_of handler))).
  { apply (ready_after_variables params r news); [ exact Hnodup | lia | ].
    intros a Ha. destruct (Houtside a Ha) as [Hv Hk]. split.
    - rewrite map_app, <- variables_of_renamed. exact Hv.
    - rewrite <- continuations_of_renamed. exact Hk. }
  cbn [rename_handler apart_handler].
  destruct (bind_variables r news params) as [[rest params'] r'].
  cbn [fst snd] in Hrest, Htaken, Hunbound, Hconts, Hbound, Hready.
  subst rest params'.
  rewrite <- variables_of_renamed, <- continuations_of_renamed in Hready.
  specialize (IH r' (skipn (length params) news) Hclosed Hdistinct Hready).
  destruct (apart r' (skipn (length params) news) handler)
    as [rest' handler'].
  cbn [snd] in IH |- *.
  (* The rule for the parameters, at the renaming that undoes [r] and
     does [r']. *)
  assert (Hmoved :
    rename_handler (moved r r')
      (Handler (map (rename_variable r) params) metadata
         (rename_expr r handler))
    = Handler (firstn (length params) news) metadata (rename_expr r' handler)).
  { change
      (rename_handler (moved r r')
         (rename_handler r (Handler params metadata handler))
       = Handler (firstn (length params) news) metadata
           (rename_expr r' handler)).
    rewrite moved_handler. cbn [rename_handler]. rewrite Hbound.
    reflexivity. }
  assert (Hcode : code_closed (rename_expr r handler) = true)
    by (rewrite code_closed_renamed; exact Hclosed).
  assert (Hfixes :
    fixes_names (moved r r')
      (free_variables_of_handler
         (Handler (map (rename_variable r) params) metadata
            (rename_expr r handler)))
      (free_continuations_of_expr (rename_expr r handler))).
  { split.
    - intros y Hy. cbn [free_variables_of_handler] in Hy.
      apply among_without_in in Hy as [Hfree Hnot].
      change (rename_variable (moved r r') y = y).
      pose proof (variable_renamed r y) as Hy0.
      remember (rename_variable (inverse_renaming r) y) as y0 eqn:Hdef.
      clear Hdef. subst y. apply moved_fixes_variable, Hunbound;
        [ exact Hlength | | ].
      + intros Hin. apply Hnot, in_map, Hin.
      + intros Hin.
        apply (proj1 (Houtside _ (in_firstn _ _ news Hin))), in_or_app.
        right. apply free_variables_in_variables_of, Hfree.
    - intros k _. change (rename_continuation (moved r r') k = k).
      pose proof (continuation_renamed r k) as Hk0.
      remember (rename_continuation (inverse_renaming r) k) as k0 eqn:Hdef.
      clear Hdef. subst k. apply moved_fixes_continuation.
      unfold rename_continuation. rewrite Hconts. reflexivity. }
  split.
  - intros k body.
    apply Alpha_trans
      with (Let_cont_non_recursive k
              (Handler (firstn (length params) news) metadata
                 (rename_expr r' handler))
              body).
    + rewrite <- Hmoved. exact (Alpha_handler _ k _ _ _ body Hcode Hfixes).
    + exact
        (Alpha_context
           (Let_cont_non_recursive_handler k (firstn (length params) news)
              metadata Hole body)
           _ _ IH).
  - intros before k after body.
    apply Alpha_trans
      with (Let_cont_recursive
              (before
               ++ (k,
                   Handler (firstn (length params) news) metadata
                     (rename_expr r' handler))
               :: after)
              body).
    + rewrite <- Hmoved.
      exact (Alpha_recursive_handler _ before k _ _ _ after body Hcode Hfixes).
    + exact
        (Alpha_context
           (Let_cont_recursive_handler before k
              (firstn (length params) news) metadata Hole after body)
           _ _ IH).
Qed.

(* ======================================================================== *)
(* A non-recursive [Let_cont]                                               *)
(* ======================================================================== *)

Lemma let_cont_apart_alpha (k : continuation) (h : continuation_handler)
    (body : expr) :
  apart_handler_alpha h -> apart_alpha body ->
  apart_alpha (Let_cont_non_recursive k h body).
Proof.
  intros IHh IHbody r news Hclosed Hdistinct (Hnodup & Houtside & Hcount).
  cbn [code_closed binders_distinct binders] in Hclosed, Hdistinct, Hcount.
  apply andb_true_iff in Hclosed as [Hclosed_h Hclosed_body].
  apply andb_true_iff in Hdistinct as [Hdistinct_h Hdistinct_body].
  cbn [rename_expr variables_of continuations_of] in Houtside.
  (* The handler. *)
  assert (Hready_h :
    ready news (binders_in_handler h)
      (variables_in_handler (rename_handler r h))
      (continuations_in_handler (rename_handler r h))).
  { split; [ exact Hnodup | ]. split; [ | lia ].
    intros a Ha. destruct (Houtside a Ha) as [Hv Hk]. split.
    - intros Hin. apply Hv, in_or_app. left. exact Hin.
    - intros Hin. apply Hk. right. apply in_or_app. left. exact Hin. }
  specialize (IHh r news Hclosed_h Hdistinct_h Hready_h).
  pose proof (apart_handler_rest h r news) as Hrest.
  cbn [rename_expr apart].
  destruct (apart_handler r news h) as [rest h'].
  cbn [fst snd] in Hrest, IHh. subst rest.
  (* The new name of the continuation. *)
  pose proof (length_skipn (binders_in_handler h) news) as Hleft.
  pose proof (NoDup_skipn (binders_in_handler h) news Hnodup) as Hnodup'.
  assert (Hin :
    ∀ b : positive, In b (skipn (binders_in_handler h) news) -> In b news)
    by (intros b; apply in_skipn).
  destruct (skipn (binders_in_handler h) news) as [ | a rest];
    [ simpl in Hleft; lia | ].
  cbn [new_continuation hd tl].
  assert (Ha : In a news) by (apply Hin; left; reflexivity).
  assert (Hready :
    ready rest (binders body)
      (variables_of (rename_expr (bind_continuation r k a) body))
      (continuations_of (rename_expr (bind_continuation r k a) body))).
  { rewrite variables_of_renamed, continuations_of_renamed.
    refine
      (ready_after_continuations [k] r (a :: rest) (binders body) _ _ Hnodup'
         _ _);
      [ simpl in Hleft |- *; lia | ].
    intros b Hb. destruct (Houtside b (Hin b Hb)) as [Hv Hk]. split.
    - intros Hmap. apply Hv, in_or_app. right.
      rewrite variables_of_renamed. exact Hmap.
    - intros Hmap. apply Hk. cbn [app map] in Hmap.
      destruct Hmap as [Hmap | Hmap]; [ left; exact Hmap | ].
      right. apply in_or_app. right. rewrite continuations_of_renamed.
      exact Hmap. }
  specialize
    (IHbody (bind_continuation r k a) rest Hclosed_body Hdistinct_body Hready).
  destruct (apart (bind_continuation r k a) rest body) as [rest' body'].
  cbn [snd] in IHbody |- *.
  apply Alpha_trans
    with (Let_cont_non_recursive (rename_continuation r k) h'
            (rename_expr r body));
    [ exact (proj1 IHh _ _) | ].
  apply Alpha_trans
    with (Let_cont_non_recursive a h'
            (rename_expr (bind_continuation r k a) body)).
  - pose proof
      (Alpha_let_cont (swap_continuation (rename_continuation r k) a)
         (rename_continuation r k) h' (rename_expr r body))
      as Hstep.
    rewrite swap_continuation_first, <- rename_expr_compose_aux in Hstep.
    apply Hstep; [ rewrite code_closed_renamed; exact Hclosed_body | ].
    split; [ intros y _; reflexivity | ].
    intros j Hj. apply among_without_in in Hj as [Hfree Hbound].
    apply swap_other.
    + intros Heq. apply Hbound. left. symmetry. exact Heq.
    + intros Heq. apply (proj2 (Houtside a Ha)). right. apply in_or_app.
      right. apply free_continuations_in_continuations_of. rewrite <- Heq.
      exact Hfree.
  - exact (Alpha_context (Let_cont_non_recursive_body a h' Hole) _ _ IHbody).
Qed.

(* ======================================================================== *)
(* A recursive [Let_cont]                                                   *)
(* ======================================================================== *)

(* One handler after another: [before] is what has been replaced. *)
Lemma handlers_replaced (hs hs' : list continuation_handler) (body : expr) :
  Forall2 handler_replaceable hs hs' ->
  ∀ (ks : list continuation)
    (before : list (continuation * continuation_handler)),
    alpha_equivalent (Let_cont_recursive (before ++ combine ks hs) body)
      (Let_cont_recursive (before ++ combine ks hs') body).
Proof.
  induction 1 as [ | h h' hs hs' Hh _ IH]; intros ks before.
  - destruct ks; apply Alpha_refl.
  - destruct ks as [ | k ks]; [ apply Alpha_refl | ]. cbn [combine].
    apply Alpha_trans
      with (Let_cont_recursive (before ++ (k, h') :: combine ks hs) body);
      [ exact (proj2 Hh before k (combine ks hs) body) | ].
    pose proof (IH ks (before ++ [(k, h')])) as Hrest.
    rewrite <- !app_assoc in Hrest. exact Hrest.
Qed.

(* The handlers of a group, renamed, are their continuations renamed
   with their handlers renamed. *)
Lemma rename_handlers_combine (r : renaming)
    (handlers : list (continuation * continuation_handler)) :
  rename_handlers r handlers
  = combine (map (rename_continuation r) (map fst handlers))
      (map (fun kh => rename_handler r (snd kh)) handlers).
Proof.
  unfold rename_handlers.
  induction handlers as [ | [k h] handlers IH]; [ reflexivity | ].
  cbn [map fst snd combine]. rewrite IH. reflexivity.
Qed.

Lemma moved_handlers (r r' : renaming)
    (handlers : list (continuation * continuation_handler)) :
  rename_handlers (moved r r') (rename_handlers r handlers)
  = rename_handlers r' handlers.
Proof.
  unfold rename_handlers. rewrite map_map. apply map_ext. intros [k h].
  rewrite moved_continuation, moved_handler. reflexivity.
Qed.

(* The handlers of a group, each given its new names in turn, are
   replaceable one by one. *)
Lemma thread_handlers_alpha (r : renaming)
    (handlers : list (continuation * continuation_handler)) :
  Forall (fun kh => apart_handler_alpha (snd kh)) handlers ->
  forallb (fun '(_, h) => handler_code_closed h) handlers = true ->
  forallb (fun '(_, h) => handler_binders_distinct h) handlers = true ->
  ∀ news : list positive,
    NoDup news ->
    (∀ (a : positive) (kh : continuation * continuation_handler),
      In a news -> In kh handlers ->
      ~ In a (variables_in_handler (rename_handler r (snd kh)))
      ∧ ~ In a (continuations_in_handler (rename_handler r (snd kh)))) ->
    list_sum (map (fun kh => binders_in_handler (snd kh)) handlers)
    <= length news ->
    Forall2 handler_replaceable
      (map (fun kh => rename_handler r (snd kh)) handlers)
      (snd
         (thread (fun news kh => apart_handler r news (snd kh)) news
            handlers)).
Proof.
  induction 1 as [ | [k h] handlers Hh _ IH];
    intros Hclosed Hdistinct news Hnodup Houtside Hcount;
    [ apply Forall2_nil | ].
  cbn [forallb map snd] in Hclosed, Hdistinct, Hcount.
  rewrite list_sum_cons in Hcount.
  apply andb_true_iff in Hclosed as [Hclosed_h Hclosed].
  apply andb_true_iff in Hdistinct as [Hdistinct_h Hdistinct].
  cbn [thread map snd].
  assert (Hready :
    ready news (binders_in_handler h)
      (variables_in_handler (rename_handler r h))
      (continuations_in_handler (rename_handler r h))).
  { split; [ exact Hnodup | ]. split; [ | lia ].
    intros a Ha. exact (Houtside a (k, h) Ha (or_introl eq_refl)). }
  specialize (Hh r news Hclosed_h Hdistinct_h Hready). cbn [snd] in Hh.
  pose proof (apart_handler_rest h r news) as Hrest.
  destruct (apart_handler r news h) as [rest h'].
  cbn [fst snd] in Hrest, Hh. subst rest.
  assert (Hrest :
    Forall2 handler_replaceable
      (map (fun kh => rename_handler r (snd kh)) handlers)
      (snd
         (thread (fun news kh => apart_handler r news (snd kh))
            (skipn (binders_in_handler h) news) handlers))).
  { apply (IH Hclosed Hdistinct); [ apply NoDup_skipn, Hnodup | | ].
    - intros a kh Ha Hkh.
      exact (Houtside a kh (in_skipn a _ news Ha) (or_intror Hkh)).
    - rewrite length_skipn. lia. }
  destruct
    (thread (fun news kh => apart_handler r news (snd kh))
       (skipn (binders_in_handler h) news) handlers)
    as [rest' hs'].
  cbn [snd] in Hrest |- *.
  apply Forall2_cons; [ exact Hh | exact Hrest ].
Qed.

Lemma let_cont_recursive_apart_alpha
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  Forall (fun kh => apart_handler_alpha (snd kh)) handlers ->
  apart_alpha body ->
  apart_alpha (Let_cont_recursive handlers body).
Proof.
  intros IHhandlers IHbody r news Hclosed Hdistinct
    (Hnodup & Houtside & Hcount).
  pose proof (code_closed_renamed r (Let_cont_recursive handlers body))
    as Hclosed_r.
  rewrite Hclosed in Hclosed_r.
  cbn [code_closed binders_distinct binders] in Hclosed, Hdistinct, Hcount.
  apply andb_true_iff in Hclosed as [Hclosed_hs Hclosed_body].
  apply andb_true_iff in Hdistinct as [Hdistinct Hdistinct_body].
  apply andb_true_iff in Hdistinct as [Hdistinct Hdistinct_hs].
  apply andb_true_iff in Hdistinct as [_ Hgroup].
  apply distinct_NoDup in Hgroup.
  change (rename_expr r (Let_cont_recursive handlers body))
    with (Let_cont_recursive (rename_handlers r handlers)
            (rename_expr r body))
    in Hclosed_r, Houtside |- *.
  cbn [variables_of continuations_of] in Houtside.
  set (ks := map fst handlers) in *.
  set (sum := list_sum (map (fun kh => binders_in_handler (snd kh)) handlers))
    in *.
  assert (Hks : length ks = length handlers) by apply length_map.
  assert (Hlength : length ks <= length news) by lia.
  (* The names of the renamed term: those of its handlers and its body,
     and the continuations of the group. *)
  assert (Hgroup_names :
    ∀ j : continuation,
      In j (map (rename_continuation r) ks) ->
      In j
        (flat_map (fun kh => fst kh :: continuations_in_handler (snd kh))
           (rename_handlers r handlers)
         ++ continuations_of (rename_expr r body))).
  { intros j Hj. apply in_or_app. left. unfold ks in Hj.
    rewrite <- map_fst_renamed in Hj. apply in_map_iff in Hj.
    destruct Hj as (kh & <- & Hkh). apply in_flat_map. exists kh.
    split; [ exact Hkh | left; reflexivity ]. }
  assert (Hhandler_names :
    ∀ (kh : continuation * continuation_handler) (a : positive),
      In kh handlers ->
      (In a (map (rename_variable r) (variables_in_handler (snd kh))) ->
       In a
         (flat_map (fun kh => variables_in_handler (snd kh))
            (rename_handlers r handlers)
          ++ variables_of (rename_expr r body)))
      ∧ (In a
           (map (rename_continuation r) (continuations_in_handler (snd kh)))
         ->
         In a
           (flat_map (fun kh => fst kh :: continuations_in_handler (snd kh))
              (rename_handlers r handlers)
            ++ continuations_of (rename_expr r body)))).
  { intros [k h] a Hkh. cbn [snd].
    assert (Hin :
      In (rename_continuation r k, rename_handler r h)
        (rename_handlers r handlers)).
    { unfold rename_handlers.
      exact
        (in_map
           (fun '(k, h) => (rename_continuation r k, rename_handler r h))
           handlers (k, h) Hkh). }
    split; intros Ha; apply in_or_app; left; apply in_flat_map;
      exists (rename_continuation r k, rename_handler r h);
      (split; [ exact Hin | ]); cbn [fst snd].
    - rewrite variables_in_handler_renamed. exact Ha.
    - right. rewrite continuations_in_handler_renamed. exact Ha. }
  (* What the renaming with the continuations of the group bound does. *)
  pose proof (bind_continuations_rest ks r news) as Hrest.
  pose proof (bind_continuations_taken ks r news Hlength) as Htaken.
  pose proof (bind_continuations_unbound ks r news) as Hunbound.
  pose proof (bind_continuations_variables ks r news) as Hvars.
  assert (Hbound :
    map (rename_continuation (snd (bind_continuations r news ks))) ks
    = firstn (length ks) news).
  { apply (bind_continuations_bound ks r news Hlength Hgroup);
      [ apply NoDup_firstn, Hnodup | ].
    intros a Ha Hin.
    exact
      (proj2 (Houtside a (in_firstn a _ news Ha)) (Hgroup_names a Hin)). }
  assert (Hready :
    ready (skipn (length ks) news) (sum + binders body)
      (map (rename_variable (snd (bind_continuations r news ks)))
         (flat_map (fun kh => variables_in_handler (snd kh)) handlers
          ++ variables_of body))
      (map (rename_continuation (snd (bind_continuations r news ks)))
         (flat_map (fun kh => continuations_in_handler (snd kh)) handlers
          ++ continuations_of body))).
  { apply (ready_after_continuations ks r news); [ exact Hnodup | lia | ].
    intros a Ha. destruct (Houtside a Ha) as [Hv Hk]. split.
    - intros Hin. apply Hv. rewrite map_app in Hin.
      apply in_app_or in Hin. destruct Hin as [Hin | Hin].
      + apply in_map_iff in Hin. destruct Hin as (y & <- & Hy).
        apply in_flat_map in Hy. destruct Hy as (kh & Hkh & Hy).
        apply (proj1 (Hhandler_names kh _ Hkh)), in_map, Hy.
      + apply in_or_app. right. rewrite variables_of_renamed. exact Hin.
    - intros Hin. apply Hk. rewrite !map_app in Hin.
      apply in_app_or in Hin. destruct Hin as [Hin | Hin];
        [ exact (Hgroup_names a Hin) | ].
      apply in_app_or in Hin. destruct Hin as [Hin | Hin].
      + apply in_map_iff in Hin. destruct Hin as (j & <- & Hj).
        apply in_flat_map in Hj. destruct Hj as (kh & Hkh & Hj).
        apply (proj2 (Hhandler_names kh _ Hkh)), in_map, Hj.
      + apply in_or_app. right. rewrite continuations_of_renamed.
        exact Hin. }
  cbn [apart]. fold ks.
  destruct (bind_continuations r news ks) as [[rest ks'] r'].
  cbn [fst snd] in Hrest, Htaken, Hunbound, Hvars, Hbound, Hready.
  subst rest ks'.
  destruct Hready as (Hnodup' & Houtside' & Hcount').
  (* The handlers, each with its new names. *)
  assert (Hhandlers :
    Forall2 handler_replaceable
      (map (fun kh => rename_handler r' (snd kh)) handlers)
      (snd
         (thread (fun news kh => apart_handler r' news (snd kh))
            (skipn (length ks) news) handlers))).
  { apply (thread_handlers_alpha r' handlers IHhandlers Hclosed_hs
             Hdistinct_hs);
      [ exact Hnodup' | | fold sum; lia ].
    intros a kh Ha Hkh. destruct (Houtside' a Ha) as [Hv Hk]. split.
    - intros Hin. apply Hv. rewrite variables_in_handler_renamed in Hin.
      apply in_map_iff in Hin. destruct Hin as (y & <- & Hy).
      apply in_map, in_or_app. left. apply in_flat_map. exists kh.
      split; assumption.
    - intros Hin. apply Hk. rewrite continuations_in_handler_renamed in Hin.
      apply in_map_iff in Hin. destruct Hin as (j & <- & Hj).
      apply in_map, in_or_app. left. apply in_flat_map. exists kh.
      split; assumption. }
  pose proof (thread_handlers_rest r' handlers (skipn (length ks) news))
    as Hrest.
  fold sum in Hrest.
  destruct
    (thread (fun news kh => apart_handler r' news (snd kh))
       (skipn (length ks) news) handlers)
    as [rest hs'].
  cbn [fst snd] in Hrest, Hhandlers. subst rest.
  (* The body. *)
  assert (Hready :
    ready (skipn sum (skipn (length ks) news)) (binders body)
      (variables_of (rename_expr r' body))
      (continuations_of (rename_expr r' body))).
  { split; [ apply NoDup_skipn, Hnodup' | ].
    split; [ | rewrite length_skipn; lia ].
    intros a Ha. apply in_skipn in Ha. destruct (Houtside' a Ha) as [Hv Hk].
    split.
    - intros Hin. apply Hv. rewrite variables_of_renamed in Hin.
      rewrite map_app. apply in_or_app. right. exact Hin.
    - intros Hin. apply Hk. rewrite continuations_of_renamed in Hin.
      rewrite map_app. apply in_or_app. right. exact Hin. }
  specialize
    (IHbody r' (skipn sum (skipn (length ks) news)) Hclosed_body
       Hdistinct_body Hready).
  destruct (apart r' (skipn sum (skipn (length ks) news)) body)
    as [rest body'].
  cbn [snd] in IHbody |- *.
  (* The continuations of the group, by the renaming that undoes [r]
     and does [r']. *)
  apply Alpha_trans
    with (Let_cont_recursive (rename_handlers r' handlers)
            (rename_expr r' body)).
  { rewrite <- (moved_handlers r r'), <- (moved_expr r r').
    apply alpha_let_cont_recursive_canonical; [ exact Hclosed_r | ].
    split.
    - intros y _. change (rename_variable (moved r r') y = y).
      pose proof (variable_renamed r y) as Hy0.
      remember (rename_variable (inverse_renaming r) y) as y0 eqn:Hdef.
      clear Hdef. subst y. apply moved_fixes_variable.
      unfold rename_variable. rewrite Hvars. reflexivity.
    - intros j Hj.
      assert (Hmember :
        In j
          (continuations_of
             (Let_cont_recursive (rename_handlers r handlers)
                (rename_expr r body)))).
      { apply free_continuations_in_continuations_of, among_In, Hj. }
      cbn [free_continuations_of_expr] in Hj.
      apply among_without_in in Hj as [Hfree Hnot].
      change (rename_continuation (moved r r') j = j).
      pose proof (continuation_renamed r j) as Hj0.
      remember (rename_continuation (inverse_renaming r) j) as j0 eqn:Hdef.
      clear Hdef. subst j. apply moved_fixes_continuation, Hunbound;
        [ exact Hlength | | ].
      + intros Hin. apply Hnot. unfold rename_handlers.
        rewrite map_fst_renamed. apply in_map. exact Hin.
      + intros Hin.
        exact (proj2 (Houtside _ (in_firstn _ _ news Hin)) Hmember). }
  (* The handlers, then the body. *)
  rewrite rename_handlers_combine. fold ks. rewrite Hbound.
  apply Alpha_trans
    with (Let_cont_recursive (combine (firstn (length ks) news) hs')
            (rename_expr r' body)).
  - exact
      (handlers_replaced _ _ (rename_expr r' body) Hhandlers
         (firstn (length ks) news) []).
  - exact
      (Alpha_context
         (Let_cont_recursive_body (combine (firstn (length ks) news) hs') Hole)
         _ _ IHbody).
Qed.

(* ======================================================================== *)
(* Code                                                                     *)
(* ======================================================================== *)

Lemma code_apart_alpha (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata) (body : expr) :
  apart_alpha body ->
  apart_code_alpha (Code params my_closure ret exn metadata body).
Proof.
  intros IH r news id Hclosed Hdistinct (Hnodup & Houtside & Hcount).
  pose proof
    (code_is_closed_renamed r
       (Code params my_closure ret exn metadata body))
    as Hclosed_r.
  rewrite Hclosed in Hclosed_r.
  cbn [code_is_closed code_binders_distinct binders_in_code]
    in Hclosed, Hdistinct, Hcount.
  apply andb_true_iff in Hclosed as [_ Hclosed].
  apply andb_true_iff in Hdistinct as [Hdistinct Hdistinct_body].
  apply andb_true_iff in Hdistinct as [Hvars Hconts].
  apply distinct_NoDup in Hvars.
  apply negb_true_iff, Pos.eqb_neq in Hconts.
  cbn [rename_code variables_in_code continuations_in_code] in Houtside.
  (* The variables that the code binds. *)
  assert (Hlength : length (my_closure :: params) <= length news)
    by (cbn [length]; lia).
  pose proof (bind_variables_rest (my_closure :: params) r news) as Hrest.
  pose proof (bind_variables_taken (my_closure :: params) r news Hlength)
    as Htaken.
  pose proof (bind_variables_continuations (my_closure :: params) r news)
    as Hsame_conts.
  assert (Hbound :
    map
      (rename_variable (snd (bind_variables r news (my_closure :: params))))
      (my_closure :: params)
    = firstn (length (my_closure :: params)) news).
  { apply
      (bind_variables_bound (my_closure :: params) r news Hlength Hvars);
      [ apply NoDup_firstn, Hnodup | ].
    intros a Ha Hin. apply (proj1 (Houtside a (in_firstn a _ news Ha))).
    cbn [map] in Hin. destruct Hin as [Hin | Hin]; [ left; exact Hin | ].
    right. apply in_or_app. left. exact Hin. }
  assert (Hready :
    ready (skipn (length (my_closure :: params)) news) (2 + binders body)
      (map
         (rename_variable
            (snd (bind_variables r news (my_closure :: params))))
         (variables_of body))
      (map
         (rename_continuation
            (snd (bind_variables r news (my_closure :: params))))
         ([ret; exn] ++ continuations_of body))).
  { apply (ready_after_variables (my_closure :: params) r news);
      [ exact Hnodup | cbn [length]; lia | ].
    intros a Ha. destruct (Houtside a Ha) as [Hv Hk]. split.
    - cbn [app map]. rewrite map_app, <- variables_of_renamed. exact Hv.
    - cbn [app map]. rewrite <- continuations_of_renamed. exact Hk. }
  cbn [rename_code apart_code].
  cbn [bind_variables] in Hrest, Htaken, Hsame_conts, Hbound, Hready.
  destruct
    (bind_variables
       (bind_variable r my_closure (new_variable r news my_closure))
       (tl news) params)
    as [[rest params'] r2].
  cbn [fst snd] in Hrest, Htaken, Hsame_conts, Hbound, Hready.
  subst rest. cbn [map] in Hbound. rewrite <- Htaken in Hbound.
  injection Hbound as Hclosure Hparams.
  destruct Hready as (Hnodup2 & Houtside2 & Hcount2).
  set (news2 := skipn (length (my_closure :: params)) news) in *.
  (* The two continuations that it binds. *)
  assert (Hlength2 : length [ret; exn] <= length news2)
    by (cbn [length]; lia).
  pose proof (bind_continuations_rest [ret; exn] r2 news2) as Hrest2.
  pose proof (bind_continuations_taken [ret; exn] r2 news2 Hlength2)
    as Htaken2.
  pose proof (bind_continuations_variables [ret; exn] r2 news2)
    as Hsame_vars.
  assert (Hbound2 :
    map (rename_continuation (snd (bind_continuations r2 news2 [ret; exn])))
      [ret; exn]
    = firstn (length [ret; exn]) news2).
  { apply (bind_continuations_bound [ret; exn] r2 news2 Hlength2);
      [ | apply NoDup_firstn, Hnodup2 | ].
    - apply NoDup_cons; [ | apply NoDup_cons; [ intros [] | apply NoDup_nil ] ].
      intros [Heq | []]. apply Hconts. symmetry. exact Heq.
    - intros a Ha Hin.
      apply (proj2 (Houtside2 a (in_firstn a _ news2 Ha))).
      rewrite map_app. apply in_or_app. left. exact Hin. }
  assert (Hready2 :
    ready (skipn (length [ret; exn]) news2) (binders body)
      (map
         (rename_variable (snd (bind_continuations r2 news2 [ret; exn])))
         (variables_of body))
      (map
         (rename_continuation
            (snd (bind_continuations r2 news2 [ret; exn])))
         (continuations_of body))).
  { apply (ready_after_continuations [ret; exn] r2 news2);
      [ exact Hnodup2 | cbn [length]; lia | exact Houtside2 ]. }
  cbn [bind_continuations fst snd length]
    in Hrest2, Htaken2, Hsame_vars, Hbound2, Hready2.
  set (ret' := new_continuation r2 news2 ret) in *.
  set (r3 := bind_continuation r2 ret ret') in *.
  set (exn' := new_continuation r3 (tl news2) exn) in *.
  set (r4 := bind_continuation r3 exn exn') in *.
  cbn [map] in Hbound2. rewrite <- Htaken2 in Hbound2.
  injection Hbound2 as Hret Hexn.
  rewrite <- Hrest2, <- variables_of_renamed, <- continuations_of_renamed
    in Hready2.
  specialize (IH r4 (tl (tl news2)) Hclosed Hdistinct_body Hready2).
  destruct (apart r4 (tl (tl news2)) body) as [rest' body'].
  cbn [snd] in IH |- *.
  (* The rule for code, at the renaming that undoes [r] and does the
     last renaming. *)
  assert (Hsame : ∀ y : variable, rename_variable r4 y = rename_variable r2 y).
  { intros y. unfold rename_variable. rewrite Hsame_vars. reflexivity. }
  assert (Hmoved :
    rename_code (moved r r4)
      (rename_code r (Code params my_closure ret exn metadata body))
    = Code params' (new_variable r news my_closure) ret' exn' metadata
        (rename_expr r4 body)).
  { rewrite moved_code. cbn [rename_code].
    rewrite (map_ext _ _ Hsame), Hsame, Hparams, Hclosure, Hret, Hexn.
    reflexivity. }
  intros before after body0.
  apply Alpha_trans
    with (Let_static
            (before
             ++ Bind_code id
                  (Code params' (new_variable r news my_closure) ret' exn'
                     metadata (rename_expr r4 body))
             :: after)
            body0).
  - rewrite <- Hmoved.
    exact
      (Alpha_code (moved r r4) before id
         (rename_code r (Code params my_closure ret exn metadata body))
         after body0 Hclosed_r).
  - exact
      (Alpha_context
         (Let_static_code before id params' (new_variable r news my_closure)
            ret' exn' metadata Hole after body0)
         _ _ IH).
Qed.

(* ======================================================================== *)
(* Static bindings                                                          *)
(* ======================================================================== *)

Lemma binding_apart_alpha (binding : static_binding) :
  match binding with
  | Bind_code _ c => apart_code_alpha c
  | Bind_deleted_code _ | Bind_symbol _ _ => True
  end ->
  apart_binding_alpha binding.
Proof.
  destruct binding as [id c | id | s const];
    intros IH r news Hclosed Hdistinct Hready;
    cbn [rename_static_binding apart_binding].
  - specialize (IH r news id Hclosed Hdistinct Hready).
    destruct (apart_code r news c) as [rest c'].
    cbn [snd] in IH |- *. exact IH.
  - intros before after body. apply Alpha_refl.
  - intros before after body. apply Alpha_refl.
Qed.

(* One binding after another: [before] is what has been replaced. *)
Lemma bindings_replaced (bindings bindings' : list static_binding)
    (body : expr) :
  Forall2 binding_replaceable bindings bindings' ->
  ∀ before : list static_binding,
    alpha_equivalent (Let_static (before ++ bindings) body)
      (Let_static (before ++ bindings') body).
Proof.
  induction 1 as [ | b b' bs bs' Hb _ IH]; intros before;
    [ apply Alpha_refl | ].
  apply Alpha_trans with (Let_static (before ++ b' :: bs) body);
    [ exact (Hb before bs body) | ].
  pose proof (IH (before ++ [b'])) as Hrest.
  rewrite <- !app_assoc in Hrest. exact Hrest.
Qed.

(* The bindings of a [Let_static], each given its new names in turn,
   are replaceable one by one. *)
Lemma thread_bindings_alpha (r : renaming) (bindings : list static_binding) :
  Forall apart_binding_alpha bindings ->
  forallb binding_code_closed bindings = true ->
  forallb binding_binders_distinct bindings = true ->
  ∀ news : list positive,
    NoDup news ->
    (∀ (a : positive) (binding : static_binding),
      In a news -> In binding bindings ->
      ~ In a (variables_in_binding (rename_static_binding r binding))
      ∧ ~ In a (continuations_in_binding (rename_static_binding r binding)))
    ->
    list_sum (map binders_in_binding bindings) <= length news ->
    Forall2 binding_replaceable (map (rename_static_binding r) bindings)
      (snd (thread (apart_binding r) news bindings)).
Proof.
  induction 1 as [ | binding bindings Hbinding _ IH];
    intros Hclosed Hdistinct news Hnodup Houtside Hcount;
    [ apply Forall2_nil | ].
  cbn [forallb map] in Hclosed, Hdistinct, Hcount.
  rewrite list_sum_cons in Hcount.
  apply andb_true_iff in Hclosed as [Hclosed_b Hclosed].
  apply andb_true_iff in Hdistinct as [Hdistinct_b Hdistinct].
  cbn [thread map].
  assert (Hready :
    ready news (binders_in_binding binding)
      (variables_in_binding (rename_static_binding r binding))
      (continuations_in_binding (rename_static_binding r binding))).
  { split; [ exact Hnodup | ]. split; [ | lia ].
    intros a Ha. exact (Houtside a binding Ha (or_introl eq_refl)). }
  specialize (Hbinding r news Hclosed_b Hdistinct_b Hready).
  pose proof (apart_binding_rest binding r news) as Hrest.
  destruct (apart_binding r news binding) as [rest binding'].
  cbn [fst snd] in Hrest, Hbinding. subst rest.
  assert (Hrest :
    Forall2 binding_replaceable (map (rename_static_binding r) bindings)
      (snd
         (thread (apart_binding r) (skipn (binders_in_binding binding) news)
            bindings))).
  { apply (IH Hclosed Hdistinct); [ apply NoDup_skipn, Hnodup | | ].
    - intros a b Ha Hb.
      exact (Houtside a b (in_skipn a _ news Ha) (or_intror Hb)).
    - rewrite length_skipn. lia. }
  destruct
    (thread (apart_binding r) (skipn (binders_in_binding binding) news)
       bindings)
    as [rest' bindings'].
  cbn [snd] in Hrest |- *.
  apply Forall2_cons; [ exact Hbinding | exact Hrest ].
Qed.

Lemma let_static_apart_alpha (bindings : list static_binding) (body : expr) :
  Forall apart_binding_alpha bindings ->
  apart_alpha body ->
  apart_alpha (Let_static bindings body).
Proof.
  intros IHbindings IHbody r news Hclosed Hdistinct
    (Hnodup & Houtside & Hcount).
  cbn [code_closed binders_distinct binders] in Hclosed, Hdistinct, Hcount.
  apply andb_true_iff in Hclosed as [Hclosed_bs Hclosed_body].
  apply andb_true_iff in Hdistinct as [Hdistinct_bs Hdistinct_body].
  cbn [rename_expr variables_of continuations_of] in Houtside.
  set (sum := list_sum (map binders_in_binding bindings)) in *.
  assert (Hbindings :
    Forall2 binding_replaceable (map (rename_static_binding r) bindings)
      (snd (thread (apart_binding r) news bindings))).
  { apply (thread_bindings_alpha r bindings IHbindings Hclosed_bs
             Hdistinct_bs);
      [ exact Hnodup | | fold sum; lia ].
    intros a binding Ha Hbinding. destruct (Houtside a Ha) as [Hv Hk].
    split; intros Hin; [ apply Hv | apply Hk ]; apply in_or_app; left;
      apply in_flat_map; exists (rename_static_binding r binding);
      (split; [ apply in_map, Hbinding | exact Hin ]). }
  pose proof (thread_bindings_rest r bindings news) as Hrest.
  fold sum in Hrest.
  cbn [rename_expr apart].
  destruct (thread (apart_binding r) news bindings) as [rest bindings'].
  cbn [fst snd] in Hrest, Hbindings. subst rest.
  assert (Hready :
    ready (skipn sum news) (binders body) (variables_of (rename_expr r body))
      (continuations_of (rename_expr r body))).
  { split; [ apply NoDup_skipn, Hnodup | ].
    split; [ | rewrite length_skipn; lia ].
    intros a Ha. apply in_skipn in Ha. destruct (Houtside a Ha) as [Hv Hk].
    split; intros Hin; [ apply Hv | apply Hk ]; apply in_or_app; right;
      exact Hin. }
  specialize (IHbody r (skipn sum news) Hclosed_body Hdistinct_body Hready).
  destruct (apart r (skipn sum news) body) as [rest body'].
  cbn [snd] in IHbody |- *.
  apply Alpha_trans with (Let_static bindings' (rename_expr r body)).
  - exact (bindings_replaced _ _ (rename_expr r body) Hbindings []).
  - exact (Alpha_context (Let_static_body bindings' Hole) _ _ IHbody).
Qed.

(* ======================================================================== *)
(* The recursion                                                            *)
(* ======================================================================== *)

Lemma apart_alpha_all :
  (∀ e : expr, apart_alpha e)
  ∧ (∀ h : continuation_handler, apart_handler_alpha h)
  ∧ (∀ binding : static_binding, apart_binding_alpha binding)
  ∧ (∀ c : code, apart_code_alpha c).
Proof.
  apply term_mutual_ind.
  - exact let_apart_alpha.
  - exact let_static_apart_alpha.
  - exact let_cont_apart_alpha.
  - exact let_cont_recursive_apart_alpha.
  - exact apply_apart_alpha.
  - exact apply_cont_apart_alpha.
  - exact switch_apart_alpha.
  - exact invalid_apart_alpha.
  - exact handler_apart_alpha.
  - exact (fun id c IH => binding_apart_alpha (Bind_code id c) IH).
  - exact (fun id => binding_apart_alpha (Bind_deleted_code id) I).
  - exact (fun s const => binding_apart_alpha (Bind_symbol s const) I).
  - exact code_apart_alpha.
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem renamed_apart_avoiding_alpha : renamed_apart_avoiding_alpha_statement.
Proof.
  intros avoid e Hclosed Hdistinct. unfold renamed_apart_avoiding.
  destruct (fresh_names_fresh (binders e) (avoid ++ names_of e))
    as [Hlength [Hnodup Houtside]].
  pose proof
    (proj1 apart_alpha_all e identity_renaming
       (fresh_names (binders e) (avoid ++ names_of e)) Hclosed Hdistinct)
    as Halpha.
  rewrite (rename_expr_identity e) in Halpha. apply Halpha.
  split; [ apply distinct_NoDup, Hnodup | ].
  split; [ | rewrite Hlength; apply le_n ].
  intros a Ha.
  assert (Hnot : among a (avoid ++ names_of e) = false)
    by (apply Houtside, among_In, Ha).
  unfold names_of in Hnot. rewrite !among_app in Hnot.
  apply orb_false_iff in Hnot as [_ Hnot].
  apply orb_false_iff in Hnot as [Hv Hk].
  split; intros Hin; apply among_In in Hin; congruence.
Qed.

Theorem renamed_apart_repeated_parameter :
  renamed_apart_repeated_parameter_statement.
Proof.
  exists
    (Let_cont_non_recursive 1
       (Handler [1; 1] (make_handler_metadata [Value; Value]) Invalid)
       Invalid)%positive.
  split; [ reflexivity | ]. split; [ reflexivity | ].
  split; [ | vm_compute; reflexivity ].
  intros Halpha. apply alpha_equivalent_binders_distinct in Halpha.
  vm_compute in Halpha. discriminate Halpha.
Qed.
