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

(* The relation between the two runs of the traversal, that of a unit and
   that of its first rebuild, for the terms of the plain class of
   FlambdaReaperUnboxingRefinesWithin4Spec.v. Nothing here needs review.

   - [sim r a1 a]: the accumulator [a1] of the run of the first rebuilt
     unit is related to [a], that of the unit, by the renaming [r]: each
     fact of [a1], renamed, is a fact of [a] or one with poison in place
     of a node of [a] ([fact_ok], [pois]); its apply deps and closure deps,
     renamed, are those of [a] (the closure deps in order, as a sublist:
     [subrel]); its symbols are those of [a]; and every made node of [a1]
     is below its counter ([below]), so that a renaming extended at the
     counter ([ext]) renames them as before.
   - Steps: both runs make a node ([sim_fresh]: the renaming takes the
     one to the other), or add facts ([sim_add]); the run of the unit
     alone does anything ([grows], [sim_grow]). [stepped r a1 r' a1' a']:
     [sim r' a1' a'], [r'] renames as [r] what is below the counter of
     [a1].
   - The environments ([env_rel]): the parameters of each continuation in
     the two runs are those of the handler, the first rebuilt unit
     keeping those of the mask of the rebuild, or nodes that the renaming
     takes to those of the unit ([ps_rel], [conts_rel]).
   - The constructs, the first rebuilt one against that of the unit:
     [step_find_cont], [step_cond_any_usage], [step_cond_alias], the call
     widgets ([step_known_widget], [step_unknown_deps],
     [step_unknown_widget]), [step_add_apply], the call kinds
     ([step_call_kind], for call kinds related by [ck_rel]), the calls
     ([step_apply], for a rebuilt call with no wrapper whose exception
     continuation keeps its first parameter), [step_apply_cont],
     [step_switch] (the arms kept are a sublist of the arms:
     [step_fold_subrel]), [step_exn_handler], the [Let]s ([step_named],
     with [step_block_deps], [step_prim_map]; [step_closure], for a set of
     one function), the poison of each as [pois].
   - [grows_traverse]: the run of the unit through a term of the class
     alone keeps the relation, for the terms that the rebuild drops.
   Not here: the induction over the terms that composes these steps, the
   code deps ([prepare_code]) and the delayed deps. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaMapSpec FlambdaMapProofs
  FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperSolutionProofs FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperUnboxingRefinesWithin2Spec
  FlambdaReaperUnboxingRefinesWithin3Spec
  FlambdaReaperUnboxingRefinesWithin4Spec.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs.
Import ListNotations.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

Definition map_fact (f : node -> node) (g : fact) : fact :=
  match g with
  | Fact_alias a b => Fact_alias (f a) (f b)
  | Fact_use a b => Fact_use (f a) (f b)
  | Fact_accessor a fl b => Fact_accessor (f a) fl (f b)
  | Fact_constructor a fl b => Fact_constructor (f a) fl (f b)
  | Fact_argument a i b => Fact_argument (f a) i (f b)
  | Fact_parameter a i b => Fact_parameter (f a) i (f b)
  | Fact_propagate a b c => Fact_propagate (f a) (f b) (f c)
  | Fact_alias_if_any_source a b c =>
    Fact_alias_if_any_source (f a) (f b) (f c)
  | Fact_any_usage a => Fact_any_usage (f a)
  | Fact_any_source a => Fact_any_source (f a)
  | Fact_any_source_if_used a b => Fact_any_source_if_used (f a) (f b)
  end.

Definition fact_nodes (g : fact) : list node :=
  match g with
  | Fact_alias a b | Fact_use a b | Fact_any_source_if_used a b => [a; b]
  | Fact_accessor a _ b | Fact_constructor a _ b | Fact_argument a _ b
  | Fact_parameter a _ b => [a; b]
  | Fact_propagate a b c | Fact_alias_if_any_source a b c => [a; b; c]
  | Fact_any_usage a | Fact_any_source a => [a]
  end.

Definition rename_fact (r : list (node * node)) (g : fact) : fact :=
  map_fact (renamed r) g.

Definition rename_ad (r : list (node * node)) (d : apply_dep) : apply_dep :=
  {| ad_function_containing_apply_expr := ad_function_containing_apply_expr d;
     ad_apply_code_id := ad_apply_code_id d;
     ad_apply_closure := renamed r (ad_apply_closure d);
     ad_apply_call_witness := renamed r (ad_apply_call_witness d) |}.

Definition rename_cld (r : list (node * node)) (d : closure_dep)
    : closure_dep :=
  {| cld_let_bound_name_of_the_closure :=
       renamed r (cld_let_bound_name_of_the_closure d);
     cld_closure_code_id := cld_closure_code_id d |}.

Definition rename_cd (r : list (node * node)) (cd : code_dep) : code_dep :=
  {| cd_params := map (renamed r) (cd_params cd);
     cd_my_closure := renamed r (cd_my_closure cd);
     cd_return := map (renamed r) (cd_return cd);
     cd_exn := renamed r (cd_exn cd);
     cd_known_arity_call_witness := renamed r (cd_known_arity_call_witness cd);
     cd_unknown_arity_call_witnesses :=
       map (renamed r) (cd_unknown_arity_call_witnesses cd) |}.

Definition cd_nodes (cd : code_dep) : list node :=
  cd_params cd ++ cd_my_closure cd :: cd_return cd ++ cd_exn cd
  :: cd_known_arity_call_witness cd :: cd_unknown_arity_call_witnesses cd.

(* A node made before the counter [k]. *)
Definition below (k : positive) (n : node) : Prop :=
  match n with
  | Node_made m => (m < k)%positive
  | Node_variable _ | Node_symbol _ | Node_code_id _ | Node_all_constants =>
    True
  end.

Definition all_below (k : positive) (l : list node) : Prop :=
  ∀ n, n ∈ l -> below k n.

(* [r'] renames the nodes below [k] as [r]. *)
Definition ext (r r' : list (node * node)) (k : positive) : Prop :=
  ∀ n, below k n -> renamed r' n = renamed r n.

Definition keys_made (r : list (node * node)) : Prop :=
  ∀ p, p ∈ r -> ∃ m, fst p = Node_made m.

Lemma below_mono (k k' : positive) (n : node) :
  (k <= k')%positive -> below k n -> below k' n.
Proof. destruct n; cbn; auto; lia. Qed.

Lemma all_below_mono (k k' : positive) (l : list node) :
  (k <= k')%positive -> all_below k l -> all_below k' l.
Proof. intros H Hl n Hn. exact (below_mono k k' n H (Hl n Hn)). Qed.

Lemma all_below_cons (k : positive) (n : node) (l : list node) :
  all_below k (n :: l) <-> below k n ∧ all_below k l.
Proof.
  split; [ intros H; split; [ apply H; left; reflexivity
                            | intros m Hm; apply H; right; exact Hm ] | ].
  intros [Hn Hl] m [<- | Hm]; [ exact Hn | exact (Hl m Hm) ].
Qed.

Lemma all_below_app (k : positive) (l l' : list node) :
  all_below k (l ++ l') <-> all_below k l ∧ all_below k l'.
Proof.
  split; [ intros H; split; intros n Hn; apply H; apply in_or_app; auto | ].
  intros [H H'] n Hn. apply in_app_or in Hn as [Hn | Hn]; auto.
Qed.

Lemma ext_refl (r : list (node * node)) (k : positive) : ext r r k.
Proof. intros n _. reflexivity. Qed.

Lemma ext_trans (r r1 r2 : list (node * node)) (k k1 : positive) :
  (k <= k1)%positive -> ext r r1 k -> ext r1 r2 k1 -> ext r r2 k.
Proof.
  intros Hk H1 H2 n Hn. rewrite (H2 n (below_mono k k1 n Hk Hn)).
  exact (H1 n Hn).
Qed.

Lemma renamed_cons_other (r : list (node * node)) (a b n : node) :
  n <> a -> renamed ((a, b) :: r) n = renamed r n.
Proof.
  intros Hne. unfold renamed. cbn [find fst].
  unfold node_eqb. destruct (node_eq_dec a n) as [-> | ]; [ congruence | ].
  reflexivity.
Qed.

Lemma renamed_cons_same (r : list (node * node)) (a b : node) :
  renamed ((a, b) :: r) a = b.
Proof.
  unfold renamed. cbn [find fst]. unfold node_eqb.
  destruct (node_eq_dec a a); [ reflexivity | congruence ].
Qed.

Lemma ext_cons (r : list (node * node)) (k : positive) (b : node) :
  ext r ((Node_made k, b) :: r) k.
Proof.
  intros n Hn. apply renamed_cons_other. intros ->. cbn in Hn. lia.
Qed.

Lemma keys_made_cons (r : list (node * node)) (k : positive) (b : node) :
  keys_made r -> keys_made ((Node_made k, b) :: r).
Proof.
  intros H p [<- | Hp]; [ exists k; reflexivity | exact (H p Hp) ].
Qed.

(* The renaming keeps a node that is no key: variables, code ids,
   symbols, the constants. *)
Lemma renamed_not_made (r : list (node * node)) (n : node) :
  keys_made r -> (∀ m, n <> Node_made m) -> renamed r n = n.
Proof.
  intros Hk Hn. unfold renamed.
  destruct (find (fun p => node_eqb (fst p) n) r) as [[a b] | ] eqn:E;
    [ | reflexivity ].
  apply find_some in E as [Hin Heq]. cbn in Heq.
  apply node_eqb_eq in Heq. subst a.
  destruct (Hk _ Hin) as (m & Hm). cbn in Hm. exfalso. exact (Hn m Hm).
Qed.

Lemma renamed_variable' (r : list (node * node)) (x : variable) :
  keys_made r -> renamed r (Node_variable x) = Node_variable x.
Proof. intros Hk. apply renamed_not_made; [ exact Hk | discriminate ]. Qed.

Lemma renamed_consts (r : list (node * node)) :
  keys_made r -> renamed r Node_all_constants = Node_all_constants.
Proof. intros Hk. apply renamed_not_made; [ exact Hk | discriminate ]. Qed.

Lemma renamed_simple (r : list (node * node)) (s : simple) :
  keys_made r -> renamed r (node_of_simple s) = node_of_simple s.
Proof.
  intros Hk. apply renamed_not_made; [ exact Hk | ].
  destruct s; discriminate.
Qed.

Lemma renamed_code_id' (r : list (node * node)) (id : code_id) :
  keys_made r -> renamed r (Node_code_id id) = Node_code_id id.
Proof. intros Hk. apply renamed_not_made; [ exact Hk | discriminate ]. Qed.

Lemma map_fact_ext (f f' : node -> node) (g : fact) :
  (∀ n, n ∈ fact_nodes g -> f n = f' n) -> map_fact f g = map_fact f' g.
Proof.
  intros H. destruct g; cbn in *;
    repeat match goal with
           | |- context [f ?n] =>
             rewrite (H n) by (repeat (left; reflexivity) || (right; left;
                               reflexivity) || (right; right; left;
                               reflexivity))
           end; reflexivity.
Qed.

Lemma rename_fact_ext (r r' : list (node * node)) (k : positive) (g : fact) :
  ext r r' k -> all_below k (fact_nodes g) ->
  rename_fact r' g = rename_fact r g.
Proof.
  intros He Hb. unfold rename_fact. apply map_fact_ext.
  intros n Hn. exact (He n (Hb n Hn)).
Qed.

Lemma map_renamed_ext (r r' : list (node * node)) (k : positive)
    (l : list node) :
  ext r r' k -> all_below k l -> map (renamed r') l = map (renamed r) l.
Proof.
  intros He Hb. apply map_ext_in. intros n Hn. exact (He n (Hb n Hn)).
Qed.

Lemma rename_cd_ext (r r' : list (node * node)) (k : positive)
    (cd : code_dep) :
  ext r r' k -> all_below k (cd_nodes cd) -> rename_cd r' cd = rename_cd r cd.
Proof.
  intros He Hb. unfold cd_nodes in Hb.
  rewrite all_below_app, all_below_cons, all_below_app, all_below_cons,
    all_below_cons in Hb.
  destruct Hb as (H1 & H2 & H3 & H4 & H5 & H6).
  unfold rename_cd. f_equal;
    [ exact (map_renamed_ext r r' k _ He H1) | exact (He _ H2)
    | exact (map_renamed_ext r r' k _ He H3) | exact (He _ H4)
    | exact (He _ H5) | exact (map_renamed_ext r r' k _ He H6) ].
Qed.

(* ======================================================================== *)
(* Facts up to poison                                                       *)
(* ======================================================================== *)

Section Pois.

Variable sol : solution.

(* [g] is the fact [f] with poison in place of a node: where the rebuild
   gives poison, the node of [f] has no use (a [Simple] rewritten), or the
   field has none (the field of a block), or the argument is deleted. *)
Inductive pois : fact -> fact -> Prop :=
  | pois_alias : ∀ a s,
      s_has_usage sol s = false ->
      pois (Fact_alias a Node_all_constants) (Fact_alias a s)
  | pois_use : ∀ a s,
      s_has_usage sol s = false ->
      pois (Fact_use a Node_all_constants) (Fact_use a s)
  | pois_constructor : ∀ b fl s,
      s_has_usage sol s = false ∨ s_field_used sol b fl = false ->
      pois (Fact_constructor b fl Node_all_constants) (Fact_constructor b fl s)
  | pois_argument : ∀ s i w,
      pois (Fact_argument Node_all_constants i w) (Fact_argument s i w)
  | pois_any_usage : ∀ s,
      s_has_usage sol s = false ->
      pois (Fact_any_usage Node_all_constants) (Fact_any_usage s).

Definition fact_ok (g : fact) (L : list fact) : Prop :=
  g ∈ L ∨ ∃ f, f ∈ L ∧ pois g f.

Lemma fact_ok_mono (g : fact) (L L' : list fact) :
  incl L L' -> fact_ok g L -> fact_ok g L'.
Proof.
  intros Hi [H | (f & Hf & Hp)]; [ left; exact (Hi _ H) | right ].
  exists f. split; [ exact (Hi _ Hf) | exact Hp ].
Qed.

Lemma fact_ok_in (g : fact) (L : list fact) : g ∈ L -> fact_ok g L.
Proof. intros H. left. exact H. Qed.

End Pois.

(* ======================================================================== *)
(* Sublists                                                                 *)
(* ======================================================================== *)

Inductive subrel {A B : Type} (R : A -> B -> Prop) : list A -> list B -> Prop :=
  | subrel_nil : subrel R [] []
  | subrel_skip : ∀ l1 l b, subrel R l1 l -> subrel R l1 (b :: l)
  | subrel_keep : ∀ a b l1 l, R a b -> subrel R l1 l -> subrel R (a :: l1) (b ::
    l).

Lemma subrel_impl {A B : Type} (R R' : A -> B -> Prop) (l1 : list A)
    (l : list B) :
  (∀ a b, a ∈ l1 -> R a b -> R' a b) -> subrel R l1 l -> subrel R' l1 l.
Proof.
  intros HR H. induction H as [ | l1 l b H IH | a b l1 l Hab H IH ].
  - constructor.
  - constructor. exact (IH HR).
  - constructor; [ apply HR; [ left; reflexivity | exact Hab ] | ].
    apply IH. intros a' b' Ha'. apply HR. right. exact Ha'.
Qed.

Lemma subrel_app_skip {A B : Type} (R : A -> B -> Prop) (l1 : list A)
    (l l' : list B) :
  subrel R l1 l -> subrel R l1 (l' ++ l).
Proof.
  intros H. induction l' as [ | b l' IH]; [ exact H | ].
  cbn. constructor. exact IH.
Qed.

Lemma subrel_in {A B : Type} (R : A -> B -> Prop) (l1 : list A) (l : list B)
    (a : A) :
  subrel R l1 l -> a ∈ l1 -> ∃ b, b ∈ l ∧ R a b.
Proof.
  intros H. induction H as [ | l1 l b H IH | a' b l1 l Hab H IH ];
    intros Ha; [ destruct Ha | | ].
  - destruct (IH Ha) as (b' & Hb & HR). exists b'. split; [ right | ]; auto.
  - destruct Ha as [<- | Ha]; [ exists b; split; [ left | ]; auto | ].
    destruct (IH Ha) as (b' & Hb & HR). exists b'. split; [ right | ]; auto.
Qed.

(* ======================================================================== *)
(* The run of the unit alone                                                *)
(* ======================================================================== *)

(* [a'] has what [a] has: more facts, apply deps and symbols, closure deps
   pushed in front, the same code deps, a counter no lower. *)
Definition grows (a a' : acc) : Prop :=
  incl (acc_facts a) (acc_facts a')
  ∧ incl (acc_apply_deps a) (acc_apply_deps a')
  ∧ (∃ l, acc_closure_deps a' = l ++ acc_closure_deps a)
  ∧ incl (acc_symbols a) (acc_symbols a')
  ∧ acc_code_deps a' = acc_code_deps a
  ∧ (acc_next a <= acc_next a')%positive.

Lemma grows_refl (a : acc) : grows a a.
Proof.
  repeat split; try apply incl_refl; [ exists []; reflexivity | lia ].
Qed.

Lemma grows_trans (a a1 a2 : acc) : grows a a1 -> grows a1 a2 -> grows a a2.
Proof.
  intros (H1 & H2 & (l & H3) & H4 & H5 & H6)
    (H1' & H2' & (l' & H3') & H4' & H5' & H6').
  repeat split; try (eapply incl_tran; eassumption); [ | congruence | lia ].
  exists (l' ++ l). rewrite H3', H3, app_assoc. reflexivity.
Qed.

Lemma grows_fresh (a : acc) : grows a (snd (fresh a)).
Proof.
  repeat split; cbn; try apply incl_refl; [ exists []; reflexivity | lia ].
Qed.

Lemma grows_add (L : list fact) (a : acc) : grows a (add_facts L a).
Proof.
  repeat split; cbn; try apply incl_refl; [ | exists []; reflexivity | lia ].
  intros g Hg. apply in_or_app. right. exact Hg.
Qed.

Lemma grows_note (s : simple) (a : acc) : grows a (note_simple s a).
Proof.
  destruct s; cbn; try apply grows_refl.
  repeat split; cbn; try apply incl_refl; [ exists []; reflexivity | | lia ].
  intros x Hx. right. exact Hx.
Qed.

Lemma grows_notes (l : list simple) (a : acc) : grows a (note_simples l a).
Proof.
  unfold note_simples. revert a.
  induction l as [ | s l IH]; intros a; [ apply grows_refl | ].
  cbn. eapply grows_trans; [ apply grows_note | apply IH ].
Qed.

Lemma grows_push_apply (d : apply_dep) (a : acc) :
  grows a (push_apply_dep d a).
Proof.
  repeat split; cbn; try apply incl_refl; [ | exists []; reflexivity | lia ].
  intros x Hx. right. exact Hx.
Qed.

Lemma grows_push_closure (d : closure_dep) (a : acc) :
  grows a (push_closure_dep d a).
Proof.
  repeat split; cbn; try apply incl_refl; [ exists [d]; reflexivity | lia ].
Qed.

Lemma grows_fresh_list (n : nat) (a : acc) : grows a (snd (fresh_list n a)).
Proof.
  revert a. induction n as [ | n IH]; intros a; [ apply grows_refl | ].
  cbn [fresh_list]. destruct (fresh a) as [x a0] eqn:Ef.
  destruct (fresh_list n a0) as [xs a'] eqn:E. cbn [snd].
  eapply grows_trans; [ | specialize (IH a0); rewrite E in IH; exact IH ].
  pose proof (grows_fresh a) as G. rewrite Ef in G. exact G.
Qed.

Lemma grows_fold {A : Type} (f : acc -> A -> acc) (l : list A) (a : acc) :
  (∀ a' x, x ∈ l -> grows a' (f a' x)) -> grows a (List.fold_left f l a).
Proof.
  revert a. induction l as [ | x l IH]; intros a H; [ apply grows_refl | ].
  cbn. eapply grows_trans; [ apply H; left; reflexivity | ].
  apply IH. intros a' y Hy. apply H. right. exact Hy.
Qed.

(* ======================================================================== *)
(* The relation                                                             *)
(* ======================================================================== *)

Section Sim.

Variable sol : solution.
(* The code deps of the two runs once the code is prepared. *)
Variables cds1 cds : finite_map code_dep.

Definition ad_nodes (d : apply_dep) : list node :=
  [ad_apply_closure d; ad_apply_call_witness d].

(* A code id of a dep that the first rebuilt unit does not prepare is not
   prepared by the unit. *)
Definition code_ok (id : code_id) : Prop :=
  lookup id cds1 = None -> lookup id cds = None.

Definition cld_rel (r : list (node * node)) (k : positive)
    (d1 d : closure_dep) : Prop :=
  below k (cld_let_bound_name_of_the_closure d1)
  ∧ code_ok (cld_closure_code_id d1) ∧ rename_cld r d1 = d.

Record sim (r : list (node * node)) (a1 a : acc) : Prop := {
  sim_facts : ∀ g, g ∈ acc_facts a1 ->
    fact_ok sol (rename_fact r g) (acc_facts a);
  sim_low : ∀ g, g ∈ acc_facts a1 -> all_below (acc_next a1) (fact_nodes g);
  sim_ads : ∀ d, d ∈ acc_apply_deps a1 ->
    all_below (acc_next a1) (ad_nodes d) ∧ code_ok (ad_apply_code_id d)
    ∧ rename_ad r d ∈ acc_apply_deps a;
  sim_cls : subrel (cld_rel r (acc_next a1)) (acc_closure_deps a1)
              (acc_closure_deps a);
  sim_syms : incl (acc_symbols a1) (acc_symbols a);
  sim_keys : keys_made r;
  sim_cds1 : acc_code_deps a1 = cds1;
  sim_cds : acc_code_deps a = cds
}.

(* A step of both runs. *)
Definition stepped (r : list (node * node)) (a1 : acc)
    (r' : list (node * node)) (a1' a' : acc) : Prop :=
  sim r' a1' a' ∧ ext r r' (acc_next a1) ∧ (acc_next a1 <= acc_next
    a1')%positive.

Lemma stepped_trans (r r1 r2 : list (node * node)) (a1 a1' a1'' a' a'' : acc) :
  stepped r a1 r1 a1' a' -> stepped r1 a1' r2 a1'' a'' ->
  stepped r a1 r2 a1'' a''.
Proof.
  intros (_ & He1 & Hk1) (Hs & He2 & Hk2).
  split; [ exact Hs | split; [ | lia ] ].
  exact (ext_trans r r1 r2 _ _ Hk1 He1 He2).
Qed.

Lemma stepped_refl (r : list (node * node)) (a1 a : acc) :
  sim r a1 a -> stepped r a1 r a1 a.
Proof. intros H. split; [ exact H | split; [ apply ext_refl | lia ] ]. Qed.

Lemma stepped_same (r : list (node * node)) (a1 a1' a' : acc) :
  sim r a1' a' -> (acc_next a1 <= acc_next a1')%positive ->
  stepped r a1 r a1' a'.
Proof. intros H Hk. split; [ exact H | split; [ apply ext_refl | exact Hk ] ].
Qed.

Lemma sim_grow (r : list (node * node)) (a1 a a' : acc) :
  sim r a1 a -> grows a a' -> sim r a1 a'.
Proof.
  intros Hs (H1 & H2 & (l & H3) & H4 & H5 & _). constructor.
  - intros g Hg. exact (fact_ok_mono _ _ _ _ H1 (sim_facts _ _ _ Hs g Hg)).
  - exact (sim_low _ _ _ Hs).
  - intros d Hd. destruct (sim_ads _ _ _ Hs d Hd) as (? & ? & ?).
    repeat split; auto.
  - rewrite H3. apply subrel_app_skip. exact (sim_cls _ _ _ Hs).
  - eapply incl_tran; [ exact (sim_syms _ _ _ Hs) | exact H4 ].
  - exact (sim_keys _ _ _ Hs).
  - exact (sim_cds1 _ _ _ Hs).
  - rewrite H5. exact (sim_cds _ _ _ Hs).
Qed.

Lemma stepped_grow (r r' : list (node * node)) (a1 a1' a' a'' : acc) :
  stepped r a1 r' a1' a' -> grows a' a'' -> stepped r a1 r' a1' a''.
Proof.
  intros (Hs & He & Hk) Hg. split; [ exact (sim_grow _ _ _ _ Hs Hg) | ].
  split; assumption.
Qed.

Lemma sim_fresh (r : list (node * node)) (a1 a : acc) :
  sim r a1 a ->
  sim ((Node_made (acc_next a1), Node_made (acc_next a)) :: r)
    (snd (fresh a1)) (snd (fresh a)).
Proof.
  intros Hs. set (r' := (Node_made (acc_next a1), Node_made (acc_next a)) :: r).
  assert (He : ext r r' (acc_next a1)) by apply ext_cons.
  constructor; cbn.
  - intros g Hg. rewrite (rename_fact_ext r r' _ g He (sim_low _ _ _ Hs g Hg)).
    exact (sim_facts _ _ _ Hs g Hg).
  - intros g Hg. apply (all_below_mono (acc_next a1)); [ lia | ].
    exact (sim_low _ _ _ Hs g Hg).
  - intros d Hd. destruct (sim_ads _ _ _ Hs d Hd) as (Hb & Hc & Hin).
    split; [ apply (all_below_mono (acc_next a1)); [ lia | exact Hb ] | ].
    split; [ exact Hc | ].
    replace (rename_ad r' d) with (rename_ad r d); [ exact Hin | ].
    unfold rename_ad. rewrite (He _ (Hb _ (or_introl eq_refl))),
      (He _ (Hb _ (or_intror (or_introl eq_refl)))). reflexivity.
  - eapply subrel_impl; [ | exact (sim_cls _ _ _ Hs) ].
    intros d1 d _ (Hb & Hc & Hr). split; [ | split; [ exact Hc | ] ].
    + apply (below_mono (acc_next a1)); [ cbn; lia | exact Hb ].
    + rewrite <- Hr. unfold rename_cld. rewrite (He _ Hb). reflexivity.
  - exact (sim_syms _ _ _ Hs).
  - apply keys_made_cons. exact (sim_keys _ _ _ Hs).
  - exact (sim_cds1 _ _ _ Hs).
  - exact (sim_cds _ _ _ Hs).
Qed.

Lemma stepped_fresh (r : list (node * node)) (a1 a : acc) :
  sim r a1 a ->
  stepped r a1 ((Node_made (acc_next a1), Node_made (acc_next a)) :: r)
    (snd (fresh a1)) (snd (fresh a)).
Proof.
  intros Hs. split; [ exact (sim_fresh r a1 a Hs) | split ].
  - apply ext_cons.
  - cbn. lia.
Qed.

Lemma sim_add (r : list (node * node)) (a1 a : acc) (L1 L : list fact) :
  sim r a1 a ->
  (∀ g, g ∈ L1 ->
     fact_ok sol (rename_fact r g) (L ++ acc_facts a)
     ∧ all_below (acc_next a1) (fact_nodes g)) ->
  sim r (add_facts L1 a1) (add_facts L a).
Proof.
  intros Hs HL. constructor; cbn.
  - intros g Hg. apply in_app_or in Hg as [Hg | Hg]; [ exact (proj1 (HL g Hg)) |
    ].
    refine (fact_ok_mono _ _ _ _ _ (sim_facts _ _ _ Hs g Hg)).
    intros x Hx. apply in_or_app. right. exact Hx.
  - intros g Hg. apply in_app_or in Hg as [Hg | Hg];
      [ exact (proj2 (HL g Hg)) | exact (sim_low _ _ _ Hs g Hg) ].
  - exact (sim_ads _ _ _ Hs).
  - exact (sim_cls _ _ _ Hs).
  - exact (sim_syms _ _ _ Hs).
  - exact (sim_keys _ _ _ Hs).
  - exact (sim_cds1 _ _ _ Hs).
  - exact (sim_cds _ _ _ Hs).
Qed.

(* The symbols that [note_simples] records. *)
Definition simple_symbols (l : list simple) : list symbol :=
  flat_map (fun s => match s with Symbol sym => [sym] | _ => [] end) l.

Lemma note_simples_fields (l : list simple) (a : acc) :
  acc_next (note_simples l a) = acc_next a
  ∧ acc_facts (note_simples l a) = acc_facts a
  ∧ acc_code_deps (note_simples l a) = acc_code_deps a
  ∧ acc_apply_deps (note_simples l a) = acc_apply_deps a
  ∧ acc_closure_deps (note_simples l a) = acc_closure_deps a
  ∧ (∀ sym, sym ∈ acc_symbols (note_simples l a)
            <-> sym ∈ acc_symbols a ∨ sym ∈ simple_symbols l).
Proof.
  unfold note_simples. revert a.
  induction l as [ | s l IH]; intros a.
  - cbn. repeat split; auto. intros [H | []]. exact H.
  - cbn [List.fold_left]. destruct (IH (note_simple s a)) as (H1 & H2 & H3 & H4
    & H5 & H6).
    rewrite H1, H2, H3, H4, H5.
    destruct s as [x | sym0 | c]; cbn [note_simple];
      refine (conj eq_refl (conj eq_refl (conj eq_refl (conj eq_refl
                (conj eq_refl _))))); intros sym; rewrite H6;
      cbn [simple_symbols flat_map acc_symbols]; cbn; tauto.
Qed.

Lemma acc_next_note_simples (l : list simple) (a : acc) :
  acc_next (note_simples l a) = acc_next a.
Proof. exact (proj1 (note_simples_fields l a)). Qed.

Lemma acc_next_add_facts (L : list fact) (a : acc) :
  acc_next (add_facts L a) = acc_next a.
Proof. reflexivity. Qed.

Lemma acc_next_cond_alias (env : traverse_env) (x y : node) (a : acc) :
  acc_next (cond_alias env x y a) = acc_next a.
Proof. unfold cond_alias. destruct (env_current_code_id env); reflexivity. Qed.

Lemma sim_notes (r : list (node * node)) (a1 a : acc) (l1 l : list simple) :
  sim r a1 a -> incl (simple_symbols l1) (simple_symbols l) ->
  sim r (note_simples l1 a1) (note_simples l a).
Proof.
  intros Hs Hi.
  destruct (note_simples_fields l1 a1) as (H1 & H2 & H3 & H4 & H5 & H6).
  destruct (note_simples_fields l a) as (H1' & H2' & H3' & H4' & H5' & H6').
  constructor; rewrite ?H1, ?H2, ?H3, ?H4, ?H5, ?H1', ?H2', ?H3', ?H4', ?H5'.
  - exact (sim_facts _ _ _ Hs).
  - exact (sim_low _ _ _ Hs).
  - exact (sim_ads _ _ _ Hs).
  - exact (sim_cls _ _ _ Hs).
  - intros sym Hsym. apply H6'. apply H6 in Hsym as [Hsym | Hsym];
      [ left; exact (sim_syms _ _ _ Hs _ Hsym) | right; exact (Hi _ Hsym) ].
  - exact (sim_keys _ _ _ Hs).
  - exact (sim_cds1 _ _ _ Hs).
  - exact (sim_cds _ _ _ Hs).
Qed.

Lemma sim_note (r : list (node * node)) (a1 a : acc) (s1 s : simple) :
  sim r a1 a -> incl (simple_symbols [s1]) (simple_symbols [s]) ->
  sim r (note_simple s1 a1) (note_simple s a).
Proof. intros Hs Hi. exact (sim_notes r a1 a [s1] [s] Hs Hi). Qed.

Lemma sim_push_apply (r : list (node * node)) (a1 a : acc) (d : apply_dep) :
  sim r a1 a -> all_below (acc_next a1) (ad_nodes d) ->
  code_ok (ad_apply_code_id d) ->
  sim r (push_apply_dep d a1) (push_apply_dep (rename_ad r d) a).
Proof.
  intros Hs Hb Hc. constructor; cbn; try apply Hs.
  intros d' [<- | Hd]; [ repeat split; [ exact Hb | exact Hc | left; reflexivity
    ] | ].
  destruct (sim_ads _ _ _ Hs d' Hd) as (? & ? & ?).
  repeat split; [ assumption | assumption | right; assumption ].
Qed.

Lemma sim_push_closure (r : list (node * node)) (a1 a : acc)
    (d : closure_dep) :
  sim r a1 a -> below (acc_next a1) (cld_let_bound_name_of_the_closure d) ->
  code_ok (cld_closure_code_id d) ->
  sim r (push_closure_dep d a1) (push_closure_dep (rename_cld r d) a).
Proof.
  intros Hs Hb Hc. constructor; cbn; try apply Hs.
  constructor; [ repeat split; assumption | exact (sim_cls _ _ _ Hs) ].
Qed.

End Sim.

(* ======================================================================== *)
(* The environments                                                         *)
(* ======================================================================== *)

Abbreviation tconts := FlambdaReaperTraversal.env_conts.
Abbreviation rconts := FlambdaReaperRebuild.env_conts.

(* The parameters of a continuation in the two runs: those of a handler,
   whose rebuild keeps those of [mask], or nodes that the renaming takes to
   those of the unit. *)
Definition ps_rel (r : list (node * node)) (k : positive)
    (m : option (list bool)) (ps1 ps : list node) : Prop :=
  match m with
  | Some mask =>
    ∃ params, ps = map Node_variable params
              ∧ ps1 = map Node_variable (filter_kept mask params)
  | None => map (renamed r) ps1 = ps ∧ all_below k ps1
  end.

Definition conts_rel (r : list (node * node)) (k : positive)
    (m : option (list bool)) (o1 o : option (list node)) : Prop :=
  match o1, o with
  | Some ps1, Some ps => ps_rel r k m ps1 ps
  | None, None => m = None
  | _, _ => False
  end.

Record env_rel (masks : finite_map (list bool)) (r : list (node * node))
    (k : positive) (env1 env : traverse_env) : Prop := {
  er_conts : ∀ c, conts_rel r k (lookup c masks) (lookup c (tconts env1))
                    (lookup c (tconts env));
  er_code : env_current_code_id env1 = env_current_code_id env;
  er_pres : env_should_preserve_direct_calls env1
            = env_should_preserve_direct_calls env;
  er_slots : incl (env_value_slots_to_keep env1) (env_value_slots_to_keep env);
  er_ucids : env_unit_code_ids env1 = env_unit_code_ids env
}.

Lemma ps_rel_ext (r r' : list (node * node)) (k k' : positive)
    (m : option (list bool)) (ps1 ps : list node) :
  ext r r' k -> (k <= k')%positive -> ps_rel r k m ps1 ps ->
  ps_rel r' k' m ps1 ps.
Proof.
  intros He Hk. destruct m as [mask | ]; [ auto | ].
  intros [Hm Hb]. split; [ | exact (all_below_mono k k' _ Hk Hb) ].
  rewrite (map_renamed_ext r r' k ps1 He Hb). exact Hm.
Qed.

Lemma env_rel_ext (masks : finite_map (list bool)) (r r' : list (node * node))
    (k k' : positive) (env1 env : traverse_env) :
  ext r r' k -> (k <= k')%positive -> env_rel masks r k env1 env ->
  env_rel masks r' k' env1 env.
Proof.
  intros He Hk H. constructor; try apply H.
  intros c. pose proof (er_conts _ _ _ _ _ H c) as Hc.
  destruct (lookup c (tconts env1)), (lookup c (tconts env)); auto.
  exact (ps_rel_ext r r' k k' _ _ _ He Hk Hc).
Qed.

(* ======================================================================== *)
(* Simples rebuilt                                                          *)
(* ======================================================================== *)

(* A simple of the first rebuilt unit, at the place of [s]: [s], or a
   constant where [s] has no use. *)
Definition simple_le (sol : solution) (s1 s : simple) : Prop :=
  s1 = s
  ∨ ((∃ c, s1 = Const c) ∧ s_has_usage sol (node_of_simple s) = false).

(* An argument of a call of the first rebuilt unit: the argument, or a
   constant. *)
Definition arg_le (s1 s : simple) : Prop := s1 = s ∨ ∃ c, s1 = Const c.

Lemma rewrite_simple_le (sol : solution) (en : rebuild_env) (s : simple) :
  simple_le sol (rewrite_simple sol en s) s.
Proof.
  unfold simple_le. destruct s as [x | sym | c]; cbn; [ | | left; reflexivity ].
  - unfold is_var_used. destruct (kind_of_variable en x); try (left;
    reflexivity);
      destruct (s_has_usage sol (Node_variable x)) eqn:E;
      try (left; reflexivity); right; split; eauto.
  - unfold is_symbol_used. destruct (s_has_usage sol (Node_symbol sym)) eqn:E;
      [ left; reflexivity | right; split; eauto ].
Qed.

Lemma simple_le_arg (sol : solution) (s1 s : simple) :
  simple_le sol s1 s -> arg_le s1 s.
Proof. intros [H | (H & _)]; [ left | right ]; assumption. Qed.

Lemma simple_le_symbols (sol : solution) (s1 s : simple) :
  simple_le sol s1 s -> incl (simple_symbols [s1]) (simple_symbols [s]).
Proof.
  intros [-> | ((c & ->) & _)]; [ apply incl_refl | ].
  intros x [].
Qed.

Lemma arg_le_symbols (s1 s : simple) :
  arg_le s1 s -> incl (simple_symbols [s1]) (simple_symbols [s]).
Proof. intros [-> | (c & ->)]; [ apply incl_refl | intros x [] ]. Qed.

Lemma symbols_forall2 (R : simple -> simple -> Prop) (l1 l : list simple) :
  (∀ s1 s, R s1 s -> incl (simple_symbols [s1]) (simple_symbols [s])) ->
  Forall2 R l1 l -> incl (simple_symbols l1) (simple_symbols l).
Proof.
  intros HR H. induction H as [ | s1 s l1 l Hs H IH]; [ apply incl_refl | ].
  intros x Hx. change (s1 :: l1) with ([s1] ++ l1) in Hx.
  change (s :: l) with ([s] ++ l).
  unfold simple_symbols in *. rewrite flat_map_app in *.
  apply in_app_or in Hx as [Hx | Hx]; apply in_or_app;
    [ left; exact (HR _ _ Hs x Hx) | right; exact (IH x Hx) ].
Qed.

Lemma symbols_sublist (l1 l : list simple) :
  incl l1 l -> incl (simple_symbols l1) (simple_symbols l).
Proof.
  intros H x Hx. unfold simple_symbols in *. apply in_flat_map in Hx as
    (s & Hs & Hx). apply in_flat_map. exists s. split; [ exact (H s Hs) | ].
  exact Hx.
Qed.

(* ======================================================================== *)
(* Steps of both runs                                                       *)
(* ======================================================================== *)

(* ======================================================================== *)
(* Lists                                                                    *)
(* ======================================================================== *)

Lemma combine_seq_forall2 {A B : Type} (R : A -> B -> Prop) (l1 : list A)
    (l : list B) (k i : nat) (x1 : A) :
  Forall2 R l1 l -> (i, x1) ∈ combine (seq k (length l1)) l1 ->
  ∃ x, (i, x) ∈ combine (seq k (length l)) l ∧ R x1 x.
Proof.
  intros H. revert k. induction H as [ | y1 y l1 l Hy H IH]; intros k Hin;
    [ destruct Hin | ].
  cbn in Hin. destruct Hin as [Heq | Hin].
  - injection Heq as <- <-. exists y. split; [ left; reflexivity | exact Hy ].
  - destruct (IH (S k) Hin) as (x & Hx & HR). exists x.
    split; [ right; exact Hx | exact HR ].
Qed.

Lemma indexed_forall2 {A B : Type} (R : A -> B -> Prop) (l1 : list A)
    (l : list B) (i : nat) (x1 : A) :
  Forall2 R l1 l -> (i, x1) ∈ indexed l1 -> ∃ x, (i, x) ∈ indexed l ∧ R x1 x.
Proof. apply combine_seq_forall2. Qed.

Lemma indexed_map {A B : Type} (f : A -> B) (l : list A) :
  indexed (map f l) = map (fun p => (fst p, f (snd p))) (indexed l).
Proof.
  unfold indexed. rewrite length_map. generalize 0.
  induction l as [ | x l IH]; intros k; [ reflexivity | ].
  cbn. rewrite IH. reflexivity.
Qed.

Lemma indexed_in_map {A B : Type} (f : A -> B) (l : list A) (i : nat)
    (x : A) :
  (i, x) ∈ indexed l -> (i, f x) ∈ indexed (map f l).
Proof.
  intros H. rewrite indexed_map. apply in_map_iff. exists (i, x).
  split; [ reflexivity | exact H ].
Qed.

Lemma indexed_in (A : Type) (l : list A) (i : nat) (x : A) :
  (i, x) ∈ indexed l -> x ∈ l.
Proof. unfold indexed. intros H. exact (in_combine_r _ _ _ _ H). Qed.

Lemma forall2_firstn {A B : Type} (R : A -> B -> Prop) (n : nat)
    (l1 : list A) (l : list B) :
  Forall2 R l1 l -> Forall2 R (firstn n l1) (firstn n l).
Proof.
  intros H. revert n. induction H as [ | x1 x l1 l Hx H IH]; intros n;
    destruct n; cbn; constructor; auto.
Qed.

Lemma forall2_skipn {A B : Type} (R : A -> B -> Prop) (n : nat)
    (l1 : list A) (l : list B) :
  Forall2 R l1 l -> Forall2 R (skipn n l1) (skipn n l).
Proof.
  intros H. revert n. induction H as [ | x1 x l1 l Hx H IH]; intros n;
    destruct n; cbn; auto; constructor; auto.
Qed.

Lemma forall2_group_by_sizes {A B : Type} (R : A -> B -> Prop)
    (sizes : list nat) (l1 : list A) (l : list B) :
  Forall2 R l1 l ->
  Forall2 (Forall2 R) (group_by_sizes sizes l1) (group_by_sizes sizes l).
Proof.
  revert l1 l. induction sizes as [ | n sizes IH]; intros l1 l H;
    cbn; constructor.
  - exact (forall2_firstn R n l1 l H).
  - apply IH. exact (forall2_skipn R n l1 l H).
Qed.

Lemma forall2_groups {A B : Type} (R : A -> B -> Prop) (a : arity)
    (l1 : list A) (l : list B) :
  Forall2 R l1 l ->
  Forall2 (Forall2 R) (at_least_one_group (group_by_parameter a l1))
    (at_least_one_group (group_by_parameter a l)).
Proof.
  intros H. unfold group_by_parameter.
  pose proof (forall2_group_by_sizes R
    (map (@length flambda_kind) (unarize_per_parameter a)) l1 l H) as G.
  destruct G as [ | g1 g gs1 gs Hg G]; cbn; repeat constructor; auto.
Qed.

Lemma forall2_impl' {A B : Type} (R R' : A -> B -> Prop) (l1 : list A)
    (l : list B) :
  (∀ x1 x, R x1 x -> R' x1 x) -> Forall2 R l1 l -> Forall2 R' l1 l.
Proof.
  intros HR H. induction H; constructor; auto.
Qed.

Lemma forall2_map_renamed (r : list (node * node)) (l1 l : list node) :
  map (renamed r) l1 = l -> Forall2 (fun x1 x => renamed r x1 = x) l1 l.
Proof.
  intros <-. induction l1 as [ | x l1 IH]; constructor; auto.
Qed.

Lemma forall2_length {A B : Type} (R : A -> B -> Prop) (l1 : list A)
    (l : list B) :
  Forall2 R l1 l -> length l1 = length l.
Proof. intros H. induction H; cbn; auto. Qed.

Lemma filter_kept_true {A : Type} (mask : list bool) (l : list A) :
  forallb (fun keep => keep) mask = true -> filter_kept mask l = l.
Proof.
  revert mask. induction l as [ | x l IH]; intros mask H; [ reflexivity | ].
  destruct mask as [ | b mask]; cbn.
  - f_equal. apply (IH []). reflexivity.
  - cbn in H. apply andb_true_iff in H as [-> H]. f_equal. exact (IH _ H).
Qed.

Lemma filter_kept_nil {A : Type} (l : list A) : filter_kept [] l = l.
Proof. apply filter_kept_true. reflexivity. Qed.

Lemma filter_kept_map {A B : Type} (f : A -> B) (mask : list bool)
    (l : list A) :
  filter_kept mask (map f l) = map f (filter_kept mask l).
Proof.
  revert mask. induction l as [ | x l IH]; intros mask; [ reflexivity | ].
  destruct mask as [ | b mask]; cbn; [ f_equal; apply IH | ].
  destruct b; cbn; [ f_equal | ]; apply IH.
Qed.

Lemma in_combine_filter_kept {A B : Type} (mask : list bool) (l1 : list A)
    (l2 : list B) (x : A) (y : B) :
  (x, y) ∈ combine (filter_kept mask l1) (filter_kept mask l2) ->
  (x, y) ∈ combine l1 l2.
Proof.
  revert mask l2. induction l1 as [ | a l1 IH]; intros mask l2 H;
    [ destruct H | ].
  destruct l2 as [ | b l2].
  - assert (E : filter_kept mask (@nil B) = []) by (destruct mask; reflexivity).
    rewrite E, combine_nil in H. destruct H.
  - destruct mask as [ | [] mask]; cbn in H |- *.
    + destruct H as [H | H]; [ left; exact H | right ].
      exact (IH [] l2 H).
    + destruct H as [H | H]; [ left; exact H | right ].
      exact (IH mask l2 H).
    + right. exact (IH mask l2 H).
Qed.

Lemma in_combine_map_r {A B C : Type} (f : B -> C) (l1 : list A)
    (l2 : list B) (x : A) (y : C) :
  (x, y) ∈ combine l1 (map f l2) -> ∃ z, y = f z ∧ (x, z) ∈ combine l1 l2.
Proof.
  revert l2. induction l1 as [ | a l1 IH]; intros l2 H; [ destruct H | ].
  destruct l2 as [ | b l2]; [ destruct H | ]. cbn in H.
  destruct H as [H | H].
  - injection H as <- <-. exists b. split; [ reflexivity | left; reflexivity ].
  - destruct (IH l2 H) as (z & -> & Hz). exists z.
    split; [ reflexivity | right; exact Hz ].
Qed.

Lemma in_combine_map_l {A B C : Type} (f : A -> C) (l1 : list A)
    (l2 : list B) (x : A) (y : B) :
  (x, y) ∈ combine l1 l2 -> (f x, y) ∈ combine (map f l1) l2.
Proof.
  revert l2. induction l1 as [ | a l1 IH]; intros l2 H; [ destruct H | ].
  destruct l2 as [ | b l2]; [ destruct H | ]. cbn in H |- *.
  destruct H as [H | H]; [ injection H as <- <-; left; reflexivity | ].
  right. exact (IH l2 H).
Qed.

Lemma filter_id_in {A : Type} (p : A -> bool) (l : list A) :
  (∀ x, x ∈ l -> p x = true) -> filter p l = l.
Proof.
  induction l as [ | x l IH]; intros H; [ reflexivity | ].
  cbn. rewrite (H x (or_introl eq_refl)). f_equal.
  apply IH. intros y Hy. apply H. right. exact Hy.
Qed.

Lemma flat_map_ext_in' {A B : Type} (f g : A -> list B) (l : list A) :
  (∀ x, x ∈ l -> f x = g x) -> flat_map f l = flat_map g l.
Proof.
  induction l as [ | x l IH]; intros H; [ reflexivity | ].
  cbn. rewrite (H x (or_introl eq_refl)). f_equal.
  apply IH. intros y Hy. apply H. right. exact Hy.
Qed.

Lemma keyed_dedup_nodup {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) (keys : list K) :
  NoDup keys -> Keyed.dedup eqb keys = keys.
Proof.
  induction keys as [ | k keys IH]; intros H; [ reflexivity | ].
  inversion H as [ | ? ? Hk Hd ]. subst. cbn. rewrite (IH Hd). f_equal.
  unfold Keyed.without. apply filter_id_in. intros k' Hk'. cbn.
  rewrite orb_false_r. destruct (eqb k' k) eqn:E; [ | reflexivity ].
  apply Heqb in E. subst. contradiction.
Qed.

Lemma keyed_compact_nodup {K A : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) (l : list (K * A)) :
  NoDup (map fst l) -> Keyed.compact eqb l = l.
Proof.
  intros H. unfold Keyed.compact, Keyed.domain.
  rewrite (keyed_dedup_nodup eqb Heqb _ H).
  induction l as [ | [k v] l IH]; [ reflexivity | ].
  inversion H as [ | ? ? Hk Hd ]. subst. cbn [map fst Keyed.tabulate flat_map].
  cbn [Keyed.lookup]. rewrite (keyed_eqb_refl eqb Heqb k). cbn [app].
  f_equal. rewrite <- (IH Hd) at 2. unfold Keyed.tabulate.
  apply flat_map_ext_in'. intros k' Hk'. cbn [Keyed.lookup].
  destruct (eqb k' k) eqn:E; [ | reflexivity ].
  apply Heqb in E. subst. contradiction.
Qed.

Lemma subrel_nodup_fst {K A B : Type} (R : K * A -> K * B -> Prop)
    (l1 : list (K * A)) (l : list (K * B)) :
  (∀ p1 p, R p1 p -> fst p1 = fst p) -> subrel R l1 l ->
  NoDup (map fst l) -> NoDup (map fst l1).
Proof.
  intros HR H. induction H as [ | l1 l b H IH | a b l1 l Hab H IH ];
    intros Hd; cbn in *.
  - constructor.
  - inversion Hd. auto.
  - inversion Hd as [ | ? ? Hb Hd' ]. subst. constructor; [ | auto ].
    rewrite (HR _ _ Hab). intros Hin. apply Hb.
    apply in_map_iff in Hin as (p1 & Hp1 & Hin).
    destruct (subrel_in R l1 l p1 H Hin) as (p & Hp & HRp).
    apply in_map_iff. exists p. split; [ | exact Hp ].
    rewrite <- (HR _ _ HRp). exact Hp1.
Qed.

Lemma subrel_flat_map {A B : Type} (f : A -> option B) (l : list (Z * A)) :
  subrel (fun p1 p => fst p1 = fst p ∧ f (snd p) = Some (snd p1))
    (flat_map (fun '(d, x) => match f x with
                              | Some y => [(d, y)]
                              | None => []
                              end) l) l.
Proof.
  induction l as [ | [d x] l IH]; cbn; [ constructor | ].
  destruct (f x) as [y | ] eqn:E; cbn.
  - constructor; [ split; [ reflexivity | exact E ] | exact IH ].
  - constructor. exact IH.
Qed.

(* A primitive that makes a block. *)
Lemma prim_of_block_dec (p : prim) :
  {(∃ k, p = Variadic (Make_block k))
   ∨ (∃ m t, p = Heap (Make_block_with_identity m t))}
  + {¬ ((∃ k, p = Variadic (Make_block k))
        ∨ (∃ m t, p = Heap (Make_block_with_identity m t)))}.
Proof.
  destruct p as [u | b | [k | k] | [m t | | | ] | rp];
    try (right; intros [(k' & E) | (m' & t' & E)]; discriminate E).
  - left. left. exists k. reflexivity.
  - left. right. exists m, t. reflexivity.
Qed.

Section Steps.

Variable sol : solution.
Variables cds1 cds : finite_map code_dep.

Local Abbreviation sim := (sim sol cds1 cds).
Local Abbreviation stepped := (stepped sol cds1 cds).

Lemma step_fresh (r : list (node * node)) (a1 a a1' a' : acc)
    (x1 x : node) :
  sim r a1 a -> fresh a1 = (x1, a1') -> fresh a = (x, a') ->
  ∃ r', stepped r a1 r' a1' a' ∧ renamed r' x1 = x
        ∧ below (acc_next a1') x1.
Proof.
  intros Hs E1 E. pose proof (stepped_fresh _ _ _ _ _ _ Hs) as H.
  unfold fresh in E1, E. injection E1 as <- <-. injection E as <- <-.
  eexists. split; [ exact H | split ].
  - apply renamed_cons_same.
  - cbn. lia.
Qed.

Lemma step_fresh_list (n : nat) :
  ∀ (r : list (node * node)) (a1 a a1' a' : acc) (xs1 xs : list node),
    sim r a1 a -> fresh_list n a1 = (xs1, a1') -> fresh_list n a = (xs, a') ->
    ∃ r', stepped r a1 r' a1' a' ∧ map (renamed r') xs1 = xs
          ∧ all_below (acc_next a1') xs1.
Proof.
  induction n as [ | n IH]; intros r a1 a a1' a' xs1 xs Hs E1 E.
  - cbn in E1, E. injection E1 as <- <-. injection E as <- <-.
    exists r. split; [ apply stepped_refl; exact Hs | ].
    split; [ reflexivity | intros n Hn; destruct Hn ].
  - cbn [fresh_list] in E1, E.
    destruct (fresh a1) as [x1 b1] eqn:F1, (fresh a) as [x b] eqn:F0.
    destruct (fresh_list n b1) as [ys1 c1] eqn:G1,
      (fresh_list n b) as [ys c] eqn:G0.
    injection E1 as <- <-. injection E as <- <-.
    destruct (step_fresh r a1 a b1 b x1 x Hs F1 F0) as (r1 & Hst1 & Hx & Hb).
    destruct (IH r1 b1 b c1 c ys1 ys (proj1 Hst1) G1 G0)
      as (r2 & Hst2 & Hys & Hbs).
    exists r2. split; [ eapply stepped_trans; [ exact Hst1 | exact Hst2 ] | ].
    destruct Hst2 as (_ & He2 & Hk2). split.
    + cbn. rewrite (He2 x1 Hb), Hx, Hys. reflexivity.
    + apply all_below_cons. split; [ | exact Hbs ].
      exact (below_mono _ _ _ Hk2 Hb).
Qed.

Lemma fact_ok_here (g : fact) (L L' : list fact) :
  g ∈ L -> fact_ok sol g (L ++ L').
Proof. intros H. left. apply in_or_app. left. exact H. Qed.

Lemma fact_ok_pois_here (g f : fact) (L L' : list fact) :
  f ∈ L -> pois sol g f -> fact_ok sol g (L ++ L').
Proof.
  intros H Hp. right. exists f. split; [ apply in_or_app; left; exact H | ].
  exact Hp.
Qed.

Lemma step_find_cont (r : list (node * node)) (a1 a a1' a' : acc)
    (env1 env : traverse_env) (c : continuation) (m : option (list bool))
    (n1 n : nat) (ps1 ps : list node) :
  sim r a1 a ->
  conts_rel r (acc_next a1) m (lookup c (tconts env1)) (lookup c (tconts env))
  -> (m = None -> n1 = n) ->
  find_cont env1 c n1 a1 = (ps1, a1') -> find_cont env c n a = (ps, a') ->
  ∃ r', stepped r a1 r' a1' a' ∧ ps_rel r' (acc_next a1') m ps1 ps.
Proof.
  intros Hs Hc Hn E1 E. unfold find_cont in E1, E.
  destruct (lookup c (tconts env1)) as [q1 | ],
    (lookup c (tconts env)) as [q | ]; try contradiction.
  - injection E1 as <- <-. injection E as <- <-.
    exists r. split; [ apply stepped_refl; exact Hs | exact Hc ].
  - cbn in Hc. subst m. specialize (Hn eq_refl). subst n1.
    destruct (fresh_list n a1) as [xs1 b1] eqn:G1,
      (fresh_list n a) as [xs b] eqn:G0.
    injection E1 as <- <-. injection E as <- <-.
    destruct (step_fresh_list n r a1 a b1 b xs1 xs Hs G1 G0)
      as (r' & (Hs' & He & Hk) & Hm & Hb).
    exists r'. split; [ split; [ | split; [ exact He | exact Hk ] ] | ].
    + apply sim_add; [ exact Hs' | ]. intros g Hg.
      apply in_map_iff in Hg as (p1 & <- & Hp1). split.
      * apply fact_ok_here. rewrite <- Hm. apply in_map_iff.
        exists (renamed r' p1). split; [ reflexivity | ].
        apply in_map. exact Hp1.
      * intros x [<- | []]. exact (Hb p1 Hp1).
    + split; [ exact Hm | exact Hb ].
Qed.

Lemma step_cond_any_usage (r : list (node * node)) (a1 a : acc)
    (env1 env : traverse_env) (s1 s : simple) :
  sim r a1 a -> env_current_code_id env1 = env_current_code_id env ->
  simple_le sol s1 s ->
  sim r (cond_any_usage env1 s1 a1) (cond_any_usage env s a).
Proof.
  intros Hs Hc Hle. unfold cond_any_usage. rewrite Hc.
  pose proof (sim_note _ _ _ r a1 a s1 s Hs (simple_le_symbols _ _ _ Hle)) as
    Hn.
  pose proof (sim_keys _ _ _ _ _ _ Hs) as Hk.
  destruct (env_current_code_id env) as [id | ]; apply sim_add; auto;
    intros g [<- | []]; (split; [ | intros x [<- | [<- | []]] || intros x
      [<- | []]; destruct s1; cbn; auto ]);
    cbn [rename_fact map_fact];
    rewrite ?renamed_code_id', renamed_simple by exact Hk;
    (destruct Hle as [-> | ((c & ->) & Hu)];
      [ apply fact_ok_here; left; reflexivity
      | eapply fact_ok_pois_here; [ left; reflexivity | ] ]);
    constructor; exact Hu.
Qed.

Lemma step_cond_alias (r : list (node * node)) (a1 a : acc)
    (env1 env : traverse_env) (from1 to1 from to : node) :
  sim r a1 a -> env_current_code_id env1 = env_current_code_id env ->
  renamed r from1 = from -> renamed r to1 = to ->
  below (acc_next a1) from1 -> below (acc_next a1) to1 ->
  sim r (cond_alias env1 from1 to1 a1) (cond_alias env from to a).
Proof.
  intros Hs Hc Hf Ht Hbf Hbt. unfold cond_alias. rewrite Hc.
  pose proof (sim_keys _ _ _ _ _ _ Hs) as Hk.
  destruct (env_current_code_id env) as [id | ]; apply sim_add; auto;
    intros g [<- | []]; cbn [rename_fact map_fact];
    rewrite ?renamed_code_id', Hf, Ht by exact Hk;
    (split; [ apply fact_ok_here; left; reflexivity | ]);
    intros x Hx; cbn in Hx; intuition (subst; cbn; auto).
Qed.

Lemma argument_fact_ok (r : list (node * node)) (s1 s : simple) (i : nat)
    (w1 w : node) (L L' : list fact) :
  keys_made r -> arg_le s1 s -> renamed r w1 = w ->
  Fact_argument (node_of_simple s) i w ∈ L ->
  fact_ok sol (rename_fact r (Fact_argument (node_of_simple s1) i w1))
    (L ++ L').
Proof.
  intros Hk Hle Hw Hin. cbn [rename_fact map_fact].
  rewrite renamed_simple by exact Hk. rewrite Hw.
  destruct Hle as [-> | (c & ->)]; [ apply fact_ok_here; exact Hin | ].
  eapply fact_ok_pois_here; [ exact Hin | constructor ].
Qed.

Lemma step_known_widget (r : list (node * node)) (a1 a a1' a' : acc)
    (env1 env : traverse_env) (ap1 ap : apply_expr) (ret1 ret : list node)
    (exn1 exn w1 w : node) :
  sim r a1 a -> env_current_code_id env1 = env_current_code_id env ->
  Forall2 arg_le (ap_args ap1) (ap_args ap) ->
  map (renamed r) ret1 = ret -> all_below (acc_next a1) ret1 ->
  renamed r exn1 = exn -> below (acc_next a1) exn1 ->
  make_known_arity_apply_widget env1 ap1 ret1 exn1 a1 = (w1, a1') ->
  make_known_arity_apply_widget env ap ret exn a = (w, a') ->
  ∃ r', stepped r a1 r' a1' a' ∧ renamed r' w1 = w
        ∧ below (acc_next a1') w1.
Proof.
  intros Hs Hc Hargs Hret Hbret Hexn Hbexn E1 E.
  unfold make_known_arity_apply_widget in E1, E.
  destruct (fresh a1) as [wit1 b1] eqn:F1, (fresh a) as [wit b] eqn:F0.
  destruct (fresh b1) as [cal1 c1] eqn:G1, (fresh b) as [cal c] eqn:G0.
  destruct (fresh c1) as [an1 d1] eqn:H1, (fresh c) as [an d] eqn:H0.
  injection E1 as <- <-. injection E as <- <-.
  destruct (step_fresh r a1 a b1 b wit1 wit Hs F1 F0) as (r1 & S1 & Hw1 & Hb1).
  destruct (step_fresh r1 b1 b c1 c cal1 cal (proj1 S1) G1 G0)
    as (r2 & S2 & Hw2 & Hb2).
  destruct (step_fresh r2 c1 c d1 d an1 an (proj1 S2) H1 H0)
    as (r3 & S3 & Hw3 & Hb3).
  pose proof (stepped_trans _ _ _ _ _ _ _ _ _ _ _ S1 S2) as S12.
  pose proof (stepped_trans _ _ _ _ _ _ _ _ _ _ _ S12 S3) as S.
  destruct S as (Hs3 & He & Hk).
  destruct S2 as (_ & He2 & Hk2). destruct S3 as (_ & He3 & Hk3).
  pose proof (sim_keys _ _ _ _ _ _ Hs3) as Hkeys.
  (* The nodes, renamed by [r3]. *)
  assert (Rw : renamed r3 wit1 = wit).
  { rewrite (He3 _ (below_mono _ _ _ Hk2 Hb1)), (He2 _ Hb1). exact Hw1. }
  assert (Rc : renamed r3 cal1 = cal).
  { rewrite (He3 _ Hb2). exact Hw2. }
  assert (Rret : map (renamed r3) ret1 = ret).
  { rewrite (map_renamed_ext r r3 _ ret1 He Hbret). exact Hret. }
  assert (Rexn : renamed r3 exn1 = exn).
  { rewrite (He _ Hbexn). exact Hexn. }
  assert (Bk : (acc_next a1 <= acc_next d1)%positive) by lia.
  exists r3. split; [ | split; [ exact Hw3 | ] ].
  - split; [ | split; [ exact He | ] ].
    + apply step_cond_alias; [ | exact Hc | exact Hw3 | exact Rw | | ].
      * apply sim_add.
        -- apply sim_notes; [ exact Hs3 | ].
           apply (symbols_forall2 arg_le); [ apply arg_le_symbols | exact Hargs
             ].
        -- intros g Hg.
           destruct (note_simples_fields (ap_args ap1) d1) as (Hn & _).
           rewrite Hn.
           apply in_app_or in Hg as [Hg | Hg];
             [ | apply in_app_or in Hg as [Hg | Hg] ].
           ++ apply in_map_iff in Hg as ([i s1] & <- & Hin).
              destruct (indexed_forall2 _ _ _ _ _ Hargs Hin) as (s & Hs' & Hle).
              split.
              ** apply (argument_fact_ok r3 s1 s i wit1 wit); auto.
                 apply in_or_app. left. apply in_map_iff.
                 exists (i, s). split; [ reflexivity | exact Hs' ].
              ** intros x Hx. cbn in Hx.
                 destruct Hx as [<- | [<- | []]];
                   [ destruct s1; cbn; auto | ].
                 apply (below_mono (acc_next b1)); [ lia | exact Hb1 ].
           ++ apply in_map_iff in Hg as ([i x1] & <- & Hin). split.
              ** apply fact_ok_here. apply in_or_app. right. apply in_or_app.
                 left. cbn [rename_fact map_fact]. rewrite Rw.
                 apply in_map_iff. exists (i, renamed r3 x1).
                 split; [ reflexivity | ]. rewrite <- Rret.
                 exact (indexed_in_map (renamed r3) ret1 i x1 Hin).
              ** intros x Hx. cbn in Hx. destruct Hx as [<- | [<- | []]].
                 --- apply (below_mono (acc_next a1)); [ lia | ].
                     apply Hbret. exact (indexed_in _ _ _ _ Hin).
                 --- apply (below_mono (acc_next b1)); [ lia | exact Hb1 ].
           ++ assert (Bw : below (acc_next d1) wit1)
                by (apply (below_mono (acc_next b1)); [ lia | exact Hb1 ]).
              assert (Bc : below (acc_next d1) cal1)
                by (apply (below_mono (acc_next c1)); [ lia | exact Hb2 ]).
              destruct Hg as [<- | [<- | [<- | []]]];
                (split; [ apply fact_ok_here; apply in_or_app; right;
                          apply in_or_app; right; cbn [rename_fact map_fact];
                          rewrite ?Rw, ?Rc, ?Rexn; cbn; auto | ]);
                intros x Hx; cbn in Hx; intuition (subst; auto).
              apply (below_mono (acc_next a1)); [ lia | exact Hbexn ].
      * apply (below_mono (acc_next d1));
          [ cbn; rewrite acc_next_note_simples; lia | exact Hb3 ].
      * apply (below_mono (acc_next b1));
          [ cbn; rewrite acc_next_note_simples; lia | exact Hb1 ].
    + rewrite acc_next_cond_alias. cbn. rewrite acc_next_note_simples. lia.
  - rewrite acc_next_cond_alias. cbn. rewrite acc_next_note_simples.
    exact Hb3.
Qed.

Definition gw_rel (r : list (node * node)) (k : positive)
    (p1 p : list simple * node) : Prop :=
  Forall2 arg_le (fst p1) (fst p) ∧ renamed r (snd p1) = snd p
  ∧ below k (snd p1).

Lemma gw_rel_ext (r r' : list (node * node)) (k k' : positive)
    (p1 p : list simple * node) :
  ext r r' k -> (k <= k')%positive -> gw_rel r k p1 p -> gw_rel r' k' p1 p.
Proof.
  intros He Hk (H1 & H2 & H3). split; [ exact H1 | split ].
  - rewrite (He _ H3). exact H2.
  - exact (below_mono _ _ _ Hk H3).
Qed.

Lemma forall2_combine_gw (r : list (node * node)) (k : positive)
    (g1 g : list (list simple)) (ws1 ws : list node) :
  Forall2 (Forall2 arg_le) g1 g -> map (renamed r) ws1 = ws ->
  all_below k ws1 ->
  Forall2 (gw_rel r k) (combine g1 ws1) (combine g ws).
Proof.
  intros H. revert ws1 ws.
  induction H as [ | x1 x g1 g Hx H IH]; intros ws1 ws Hw Hb;
    [ constructor | ].
  destruct ws1 as [ | w1 ws1]; subst ws; cbn; [ constructor | ].
  constructor.
  - split; [ exact Hx | split; [ reflexivity | apply Hb; left; reflexivity ] ].
  - apply IH; [ reflexivity | intros n Hn; apply Hb; right; exact Hn ].
Qed.

Lemma step_unknown_deps (ret1 ret : list node) (exn1 exn cal1 cal : node) :
  ∀ (gw1 gw : list (list simple * node)) (r : list (node * node))
    (a1 a : acc),
    sim r a1 a -> Forall2 (gw_rel r (acc_next a1)) gw1 gw ->
    map (renamed r) ret1 = ret -> all_below (acc_next a1) ret1 ->
    renamed r exn1 = exn -> below (acc_next a1) exn1 ->
    renamed r cal1 = cal -> below (acc_next a1) cal1 ->
    ∃ r', stepped r a1 r' (unknown_arity_apply_deps ret1 exn1 cal1 gw1 a1)
            (unknown_arity_apply_deps ret exn cal gw a).
Proof.
  induction gw1 as [ | [first1 w1] rest1 IH];
    intros gw r a1 a Hs Hgw Hret Hbret Hexn Hbexn Hcal Hbcal;
    destruct gw as [ | [first w] rest];
    inversion Hgw as [ | ? ? ? ? Hfw Hrest ];
    [ exists r; apply stepped_refl; exact Hs | ].
  destruct Hfw as (Hf & Hw & Hbw). cbn [fst snd] in Hf, Hw, Hbw.
  cbn [unknown_arity_apply_deps].
  pose proof (sim_keys _ _ _ _ _ _ Hs) as Hk.
  assert (Hs1 : sim r
     (add_facts
        (map (fun '(i, s) => Fact_argument (node_of_simple s) i w1)
           (indexed first1)
         ++ [Fact_accessor exn1 (Field_return_of_call Exn) w1;
             Fact_accessor cal1 Field_code_id_of_call_witness w1]) a1)
     (add_facts
        (map (fun '(i, s) => Fact_argument (node_of_simple s) i w)
           (indexed first)
         ++ [Fact_accessor exn (Field_return_of_call Exn) w;
             Fact_accessor cal Field_code_id_of_call_witness w]) a)).
  { apply sim_add; [ exact Hs | ]. intros g Hg.
    apply in_app_or in Hg as [Hg | Hg].
    - apply in_map_iff in Hg as ([i s1] & <- & Hin).
      destruct (indexed_forall2 _ _ _ _ _ Hf Hin) as (s & Hs' & Hle).
      split.
      + apply (argument_fact_ok r s1 s i w1 w); auto.
        apply in_or_app. left. apply in_map_iff.
        exists (i, s). split; [ reflexivity | exact Hs' ].
      + intros n0 Hx. cbn in Hx.
        destruct Hx as [<- | [<- | []]]; [ destruct s1; cbn; auto | exact Hbw ].
    - destruct Hg as [<- | [<- | []]];
        (split; [ apply fact_ok_here; apply in_or_app; right;
                  cbn [rename_fact map_fact]; rewrite ?Hexn, ?Hcal, Hw; cbn;
                  auto | ]);
        intros n0 Hx; cbn in Hx; intuition (subst; auto). }
  destruct rest1 as [ | [nf1 nw1] rest1']; destruct rest as [ | [nf nw] rest0];
    inversion Hrest as [ | ? ? ? ? Hp' Hrest' ].
  - exists r. apply stepped_same; [ | cbn; lia ].
    apply sim_add; [ exact Hs1 | ].
    intros g Hg. apply in_map_iff in Hg as ([i x1] & <- & Hin). split.
    + apply fact_ok_here. cbn [rename_fact map_fact]. rewrite Hw.
      apply in_map_iff. exists (i, renamed r x1). split; [ reflexivity | ].
      rewrite <- Hret. exact (indexed_in_map (renamed r) ret1 i x1 Hin).
    + intros n0 Hx. cbn in Hx. destruct Hx as [<- | [<- | []]];
        [ apply Hbret; exact (indexed_in _ _ _ _ Hin) | ].
      exact Hbw.
  - destruct Hp' as (Hnf & Hnw & Hbnw). cbn [fst snd] in *.
    destruct (fresh (add_facts _ a1)) as [v1 b1] eqn:F1.
    destruct (fresh (add_facts _ a)) as [v b] eqn:F0.
    destruct (step_fresh _ _ _ _ _ _ _ Hs1 F1 F0) as (r1 & S1 & Hv & Hbv).
    destruct S1 as (Hs2 & He1 & Hk1). cbn [acc_next add_facts] in Hk1.
    assert (Hk' : (acc_next a1 <= acc_next b1)%positive) by exact Hk1.
    pose proof (sim_keys _ _ _ _ _ _ Hs2) as Hk2.
    assert (Hs3 : sim r1
      (add_facts
         [Fact_accessor v1 (Field_return_of_call (Normal 0)) w1;
          Fact_accessor nw1 (Field_call_witness Unknown_arity_code_pointer)
            v1] b1)
      (add_facts
         [Fact_accessor v (Field_return_of_call (Normal 0)) w;
          Fact_accessor nw (Field_call_witness Unknown_arity_code_pointer)
            v] b)).
    { apply sim_add; [ exact Hs2 | ].
      assert (Rw : renamed r1 w1 = w) by (rewrite (He1 _ Hbw); exact Hw).
      assert (Rnw : renamed r1 nw1 = nw)
        by (rewrite (He1 _ Hbnw); exact Hnw).
      intros g [<- | [<- | []]];
        (split; [ apply fact_ok_here; cbn [rename_fact map_fact];
                  rewrite Hv, ?Rw, ?Rnw; cbn; auto | ]).
      - intros n0 [<- | [<- | []]];
          [ exact Hbv | exact (below_mono _ _ _ Hk' Hbw) ].
      - intros n0 [<- | [<- | []]];
          [ exact (below_mono _ _ _ Hk' Hbnw) | exact Hbv ]. }
    destruct (IH ((nf, nw) :: rest0) r1 _ _ Hs3) as (r2 & S2).
    + apply (forall2_impl' (gw_rel r (acc_next a1))).
      * intros q1 q Hq. exact (gw_rel_ext r r1 _ _ q1 q He1 Hk' Hq).
      * constructor; [ | exact Hrest' ].
        split; [ exact Hnf | split; [ exact Hnw | exact Hbnw ] ].
    + rewrite (map_renamed_ext r r1 _ ret1 He1 Hbret). exact Hret.
    + exact (all_below_mono _ _ _ Hk' Hbret).
    + rewrite (He1 _ Hbexn). exact Hexn.
    + exact (below_mono _ _ _ Hk' Hbexn).
    + rewrite (He1 _ Hbcal). exact Hcal.
    + exact (below_mono _ _ _ Hk' Hbcal).
    + exists r2. destruct S2 as (Hs4 & He2 & Hk4).
      split; [ exact Hs4 | split ].
      * apply (ext_trans r r1 r2 (acc_next a1) (acc_next b1)); auto.
      * cbn in Hk4 |- *. lia.
Qed.

Lemma acc_next_unknown_deps (ret : list node) (exn cal : node)
    (gw : list (list simple * node)) (a : acc) :
  (acc_next a <= acc_next (unknown_arity_apply_deps ret exn cal gw a))%positive.
Proof.
  revert a. induction gw as [ | [g w] rest IH]; intros a; [ cbn; lia | ].
  cbn [unknown_arity_apply_deps].
  destruct rest as [ | [g' w'] rest']; [ cbn; lia | ].
  cbn [fresh]. cbv beta iota zeta.
  specialize (IH (add_facts
    [Fact_accessor (Node_made (acc_next a)) (Field_return_of_call (Normal 0))
       w;
     Fact_accessor w' (Field_call_witness Unknown_arity_code_pointer)
       (Node_made (acc_next a))]
    {| acc_next := Pos.succ (acc_next
         (add_facts
            (map (λ '(i, s), Fact_argument (node_of_simple s) i w)
               (indexed g)
             ++ [Fact_accessor exn (Field_return_of_call Exn) w;
                 Fact_accessor cal Field_code_id_of_call_witness w]) a));
       acc_facts := acc_facts (add_facts
            (map (λ '(i, s), Fact_argument (node_of_simple s) i w)
               (indexed g)
             ++ [Fact_accessor exn (Field_return_of_call Exn) w;
                 Fact_accessor cal Field_code_id_of_call_witness w]) a);
       acc_code_deps := acc_code_deps a;
       acc_apply_deps := acc_apply_deps a;
       acc_closure_deps := acc_closure_deps a;
       acc_symbols := acc_symbols a |})).
  cbn in IH |- *. lia.
Qed.

Lemma step_unknown_widget (r : list (node * node)) (a1 a a1' a' : acc)
    (env1 env : traverse_env) (ap1 ap : apply_expr) (ret1 ret : list node)
    (exn1 exn w1 w : node) :
  sim r a1 a -> env_current_code_id env1 = env_current_code_id env ->
  ap_args_arity ap1 = ap_args_arity ap ->
  Forall2 arg_le (ap_args ap1) (ap_args ap) ->
  map (renamed r) ret1 = ret -> all_below (acc_next a1) ret1 ->
  renamed r exn1 = exn -> below (acc_next a1) exn1 ->
  make_unknown_arity_apply_widget env1 ap1 ret1 exn1 a1 = (w1, a1') ->
  make_unknown_arity_apply_widget env ap ret exn a = (w, a') ->
  ∃ r', stepped r a1 r' a1' a' ∧ renamed r' w1 = w
        ∧ below (acc_next a1') w1.
Proof.
  intros Hs Hc Har Hargs Hret Hbret Hexn Hbexn E1 E.
  unfold make_unknown_arity_apply_widget in E1, E. rewrite Har in E1.
  pose proof (forall2_groups arg_le (ap_args_arity ap) _ _ Hargs) as Hg.
  set (g1 := at_least_one_group
               (group_by_parameter (ap_args_arity ap) (ap_args ap1))) in *.
  set (g := at_least_one_group
              (group_by_parameter (ap_args_arity ap) (ap_args ap))) in *.
  pose proof (forall2_length _ _ _ Hg) as Hlen.
  destruct (fresh a1) as [cal1 b1] eqn:F1, (fresh a) as [cal b] eqn:F0.
  destruct (step_fresh r a1 a b1 b cal1 cal Hs F1 F0)
    as (r1 & (Hs1 & He1 & Hk1) & Hcal & Hbcal).
  assert (Hs1' : sim r1 (add_fact (Fact_any_usage cal1) b1)
                   (add_fact (Fact_any_usage cal) b)).
  { apply sim_add; [ exact Hs1 | ]. intros x [<- | []].
    split; [ apply fact_ok_here; cbn; rewrite Hcal; left; reflexivity | ].
    intros n0 [<- | []]. exact Hbcal. }
  rewrite Hlen in E1.
  destruct (fresh_list (length g) (add_fact (Fact_any_usage cal1) b1))
    as [ws1 c1] eqn:G1.
  destruct (fresh_list (length g) (add_fact (Fact_any_usage cal) b))
    as [ws c] eqn:G0.
  destruct (step_fresh_list _ _ _ _ _ _ _ _ Hs1' G1 G0)
    as (r2 & (Hs2 & He2 & Hk2) & Hws & Hbws).
  cbn [acc_next add_fact add_facts] in Hk2.
  pose proof (sim_notes _ _ _ r2 c1 c (ap_args ap1) (ap_args ap) Hs2
    (symbols_forall2 arg_le _ _ arg_le_symbols Hargs)) as Hs3.
  destruct (note_simples_fields (ap_args ap1) c1) as (Hn1 & _).
  assert (Hk12 : (acc_next a1 <= acc_next c1)%positive) by (cbn in Hk1; lia).
  destruct (step_unknown_deps ret1 ret exn1 exn cal1 cal (combine g1 ws1)
              (combine g ws) r2 _ _ Hs3) as (r3 & S3).
  { rewrite Hn1. apply forall2_combine_gw; [ exact Hg | exact Hws | ].
    exact Hbws. }
  { rewrite (map_renamed_ext r r2 (acc_next a1) ret1);
      [ exact Hret | | exact Hbret ].
    exact (ext_trans r r1 r2 _ _ Hk1 He1 He2). }
  { rewrite Hn1. exact (all_below_mono _ _ _ Hk12 Hbret). }
  { rewrite (ext_trans r r1 r2 _ _ Hk1 He1 He2 _ Hbexn). exact Hexn. }
  { rewrite Hn1. exact (below_mono _ _ _ Hk12 Hbexn). }
  { rewrite (He2 _ Hbcal). exact Hcal. }
  { rewrite Hn1. exact (below_mono _ _ _ Hk2 Hbcal). }
  destruct S3 as (Hs4 & He3 & Hk3).
  set (d1 := unknown_arity_apply_deps ret1 exn1 cal1 (combine g1 ws1)
               (note_simples (ap_args ap1) c1)) in *.
  set (d := unknown_arity_apply_deps ret exn cal (combine g ws)
              (note_simples (ap_args ap) c)) in *.
  destruct (fresh d1) as [an1 e1] eqn:H1, (fresh d) as [an e0] eqn:H0.
  destruct (step_fresh r3 d1 d e1 e0 an1 an Hs4 H1 H0)
    as (r4 & (Hs5 & He4 & Hk4) & Han & Hban).
  rewrite Hn1 in Hk3, He3.
  assert (Ext : ext r r4 (acc_next a1)).
  { apply (ext_trans r r3 r4 _ (acc_next d1)); [ lia | | exact He4 ].
    apply (ext_trans r r2 r3 _ (acc_next c1)); [ lia | | exact He3 ].
    exact (ext_trans r r1 r2 _ _ Hk1 He1 He2). }
  destruct ws1 as [ | f1 ws1'];
    destruct ws as [ | f ws']; try discriminate Hws;
    injection E1 as <- <-; injection E as <- <-.
  - exists r4. split; [ split; [ exact Hs5 | split; [ exact Ext | lia ] ] | ].
    split; [ exact Han | exact Hban ].
  - injection Hws as Hf Hws'.
    assert (Bf : below (acc_next c1) f1)
      by (apply Hbws; left; reflexivity).
    exists r4. split; [ split; [ | split; [ exact Ext | ] ] | ].
    + apply step_cond_alias; [ exact Hs5 | exact Hc | exact Han | | exact Hban |
      ].
      * rewrite (He4 _ (below_mono _ _ _ Hk3 Bf)).
        rewrite (He3 _ Bf). exact Hf.
      * apply (below_mono (acc_next c1)); [ lia | exact Bf ].
    + rewrite acc_next_cond_alias. lia.
    + split; [ exact Han | ]. rewrite acc_next_cond_alias. exact Hban.
Qed.

Lemma step_add_apply (r : list (node * node)) (a1 a : acc)
    (env1 env : traverse_env) (id : code_id) (c1 c w1 w : node) (b : bool) :
  sim r a1 a -> env_current_code_id env1 = env_current_code_id env ->
  renamed r c1 = c -> below (acc_next a1) c1 ->
  renamed r w1 = w -> below (acc_next a1) w1 -> code_ok cds1 cds id ->
  ∃ r', stepped r a1 r' (add_apply env1 id c1 w1 b a1) (add_apply env id c w b
    a).
Proof.
  intros Hs Hc Hcr Hcb Hwr Hwb Hok. unfold add_apply. rewrite Hc.
  destruct b.
  - destruct (fresh a1) as [x1 b1] eqn:F1, (fresh a) as [x b0] eqn:F0.
    destruct (step_fresh r a1 a b1 b0 x1 x Hs F1 F0)
      as (r1 & (Hs1 & He1 & Hk1) & Hx & Hbx).
    destruct (fresh b1) as [y1 c1'] eqn:G1, (fresh b0) as [y c0] eqn:G0.
    destruct (step_fresh r1 b1 b0 c1' c0 y1 y Hs1 G1 G0)
      as (r2 & (Hs2 & He2 & Hk2) & Hy & Hby).
    assert (E : ext r r2 (acc_next a1)) by exact (ext_trans _ _ _ _ _ Hk1 He1
      He2).
    assert (Rc : renamed r2 c1 = c) by (rewrite (E _ Hcb); exact Hcr).
    assert (Rw : renamed r2 w1 = w) by (rewrite (E _ Hwb); exact Hwr).
    assert (Rx : renamed r2 x1 = x) by (rewrite (He2 _ Hbx); exact Hx).
    assert (Kc : (acc_next a1 <= acc_next c1')%positive) by lia.
    assert (Bc : below (acc_next c1') c1) by exact (below_mono _ _ _ Kc Hcb).
    assert (Bw : below (acc_next c1') w1) by exact (below_mono _ _ _ Kc Hwb).
    assert (Bx : below (acc_next c1') x1) by exact (below_mono _ _ _ Hk2 Hbx).
    exists r2. split; [ | split; [ exact E | cbn; lia ] ].
    pose proof (sim_keys _ _ _ _ _ _ Hs2) as Hk.
    replace (Apply_dep (env_current_code_id env) id x y)
      with (rename_ad r2 (Apply_dep (env_current_code_id env) id x1 y1))
      by (unfold rename_ad; cbn; rewrite Rx, Hy; reflexivity).
    apply sim_push_apply.
    + apply sim_add; [ exact Hs2 | ].
      intros g [<- | [<- | []]];
        (split; [ apply fact_ok_here; cbn [rename_fact map_fact];
                  rewrite Rc, ?Rx, ?Rw, ?Hy; cbn; auto | ]);
        intros n0 Hn0; cbn in Hn0; intuition (subst; assumption).
    + intros n0 [<- | [<- | []]]; [ exact Bx | exact Hby ].
    + exact Hok.
  - exists r. split; [ | split; [ apply ext_refl | cbn; lia ] ].
    replace (Apply_dep (env_current_code_id env) id c w)
      with (rename_ad r (Apply_dep (env_current_code_id env) id c1 w1))
      by (unfold rename_ad; cbn; rewrite Hcr, Hwr; reflexivity).
    apply sim_push_apply; [ exact Hs | | exact Hok ].
    intros n0 [<- | [<- | []]]; assumption.
Qed.

(* The call kind that the rebuild gives a call, as the traversal reads
   it, against that of the unit. *)
Definition ck_rel (pres : should_preserve_direct_calls) (ucids : list code_id)
    (ck1 ck : call_kind) : Prop :=
  match ck1, ck with
  | Indirect_unknown_arity, Indirect_unknown_arity => True
  | Indirect_known_arity _, (Indirect_known_arity _ | Direct _) => True
  | Direct id1, Direct id =>
    (id1 = id ∧ code_ok cds1 cds id)
    ∨ (pres = Preserve_no ∧ among id1 ucids = true)
  | Direct id1, Indirect_known_arity _ =>
    pres = Preserve_no ∧ among id1 ucids = true
  | _, _ => False
  end.

Lemma grows_add_apply (env : traverse_env) (id : code_id) (c w : node)
    (b : bool) (a : acc) :
  grows a (add_apply env id c w b a).
Proof.
  unfold add_apply. destruct b.
  - destruct (fresh a) as [x a0] eqn:F1. destruct (fresh a0) as [y a2] eqn:F2.
    eapply grows_trans; [ pose proof (grows_fresh a) as G; rewrite F1 in G;
                          exact G | ].
    eapply grows_trans; [ pose proof (grows_fresh a0) as G; rewrite F2 in G;
                          exact G | ].
    eapply grows_trans; [ apply grows_add | apply grows_push_apply ].
  - apply grows_push_apply.
Qed.

Lemma step_call_kind (r : list (node * node)) (a1 a : acc)
    (env1 env : traverse_env) (ap1 ap : apply_expr) (ret1 ret : list node)
    (exn1 exn : node) :
  sim r a1 a -> env_current_code_id env1 = env_current_code_id env ->
  env_should_preserve_direct_calls env1 = env_should_preserve_direct_calls env
  -> env_unit_code_ids env1 = env_unit_code_ids env ->
  ap_callee ap1 = ap_callee ap -> ap_args_arity ap1 = ap_args_arity ap ->
  Forall2 arg_le (ap_args ap1) (ap_args ap) ->
  map (renamed r) ret1 = ret -> all_below (acc_next a1) ret1 ->
  renamed r exn1 = exn -> below (acc_next a1) exn1 ->
  ck_rel (env_should_preserve_direct_calls env) (env_unit_code_ids env)
    (ap_call_kind ap1) (ap_call_kind ap) ->
  ∃ r', stepped r a1 r' (traverse_call_kind env1 ap1 exn1 ret1 a1)
          (traverse_call_kind env ap exn ret a).
Proof.
  intros Hs Hc Hp Hu Hcal Har Hargs Hret Hbret Hexn Hbexn Hck.
  unfold traverse_call_kind. rewrite Hcal, Hp, Hu.
  set (closure := node_of_simple (ap_callee ap)).
  pose proof (sim_note _ _ _ r a1 a (ap_callee ap) (ap_callee ap) Hs
                (incl_refl _)) as Hs0.
  set (b1 := note_simple (ap_callee ap) a1) in *.
  set (b := note_simple (ap_callee ap) a) in *.
  assert (Hb1 : acc_next b1 = acc_next a1)
    by (unfold b1; destruct (ap_callee ap); reflexivity).
  rewrite <- Hb1 in Hbret, Hbexn.
  (* The widget of known arity and the accessor of the closure. *)
  assert (Known : ∀ (k : closure_entry_point),
    ∃ r1 w1 w c1 c,
      make_known_arity_apply_widget env1 ap1 ret1 exn1 b1 = (w1, c1)
      ∧ make_known_arity_apply_widget env ap ret exn b = (w, c)
      ∧ stepped r b1 r1
          (add_fact (Fact_accessor w1 (Field_call_witness k) closure) c1)
          (add_fact (Fact_accessor w (Field_call_witness k) closure) c)
      ∧ renamed r1 w1 = w ∧ below (acc_next c1) w1).
  { intros k.
    destruct (make_known_arity_apply_widget env1 ap1 ret1 exn1 b1)
      as [w1 c1] eqn:K1.
    destruct (make_known_arity_apply_widget env ap ret exn b) as [w c] eqn:K0.
    destruct (step_known_widget r b1 b c1 c env1 env ap1 ap ret1 ret exn1 exn
                w1 w Hs0 Hc Hargs Hret Hbret Hexn Hbexn K1 K0)
      as (r1 & (Hs1 & He1 & Hk1) & Hw & Hbw).
    exists r1, w1, w, c1, c. split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ | split; [ exact Hw | exact Hbw ] ].
    split; [ | split; [ exact He1 | cbn; exact Hk1 ] ].
    apply sim_add; [ exact Hs1 | ]. intros g [<- | []].
    pose proof (sim_keys _ _ _ _ _ _ Hs1) as Hk.
    split; [ apply fact_ok_here; cbn [rename_fact map_fact]; unfold closure;
             rewrite Hw, renamed_simple by exact Hk; left; reflexivity | ].
    intros n0 [<- | [<- | []]]; [ exact Hbw | unfold closure ].
    destruct (ap_callee ap); cbn; auto. }
  assert (Hck' := Hck).
  destruct (ap_call_kind ap1) as [id1 | | ids1] eqn:Ck1,
    (ap_call_kind ap) as [id | | ids] eqn:Ck; cbn in Hck; try contradiction.
  - (* Direct, Direct *)
    destruct (Known Known_arity_code_pointer)
      as (r1 & w1 & w & c1 & c & -> & -> & S1 & Hw & Hbw).
    destruct S1 as (Hs1 & He1 & Hk1).
    set (d1 := add_fact _ c1) in *. set (d := add_fact _ c) in *.
    assert (Hbw' : below (acc_next d1) w1) by exact Hbw.
    assert (Hcl : renamed r1 closure = closure)
      by (unfold closure; apply renamed_simple; exact (sim_keys _ _ _ _ _ _
        Hs1)).
    assert (Hbcl : below (acc_next d1) closure)
      by (unfold closure; destruct (ap_callee ap); cbn; auto).
    destruct Hck as [[<- Hok] | [Hpn Hin]].
    + assert (Step : ∀ b0, ∃ r2, stepped r1 d1 r2 (add_apply env1 id1 closure w1
      b0 d1)
                     (add_apply env id1 closure w b0 d))
        by (intros b0; exact (step_add_apply r1 d1 d env1 env id1 closure
                                closure w1 w b0 Hs1 Hc Hcl Hbcl Hw Hbw' Hok)).
      destruct (env_should_preserve_direct_calls env);
        [ destruct (Step false) as (r2 & S2)
        | | destruct (Step true) as (r2 & S2) ].
      * exists r2. eapply stepped_trans; [ | exact S2 ].
        split; [ exact Hs1 | split; [ | rewrite <- Hb1; exact Hk1 ] ].
        rewrite <- Hb1. exact He1.
      * destruct (negb (among id1 (env_unit_code_ids env))).
        -- destruct (Step false) as (r2 & S2). exists r2.
           eapply stepped_trans; [ | exact S2 ].
           split; [ exact Hs1 | split; [ | rewrite <- Hb1; exact Hk1 ] ].
           rewrite <- Hb1. exact He1.
        -- exists r1. split; [ exact Hs1 | split; [ | rewrite <- Hb1; exact Hk1
          ] ].
           rewrite <- Hb1. exact He1.
      * exists r2. eapply stepped_trans; [ | exact S2 ].
        split; [ exact Hs1 | split; [ | rewrite <- Hb1; exact Hk1 ] ].
        rewrite <- Hb1. exact He1.
    + rewrite Hpn, Hin. cbn [negb]. exists r1.
      split; [ | split; [ rewrite <- Hb1; exact He1 | rewrite <- Hb1; exact Hk1
        ] ].
      destruct (negb (among id (env_unit_code_ids env)));
        [ exact (sim_grow _ _ _ _ _ _ _ Hs1 (grows_add_apply _ _ _ _ _ _))
        | exact Hs1 ].
  - (* Direct, known arity *)
    destruct Hck as [Hpn Hin]. rewrite Hpn, Hin. cbn [negb].
    destruct (Known Known_arity_code_pointer)
      as (r1 & w1 & w & c1 & c & -> & -> & S1 & Hw & Hbw).
    exists r1. destruct S1 as (Hs1 & He1 & Hk1).
    split; [ exact Hs1 | split; [ rewrite <- Hb1; exact He1 | rewrite <- Hb1;
      exact Hk1 ] ].
  - (* unknown, unknown *)
    destruct (make_unknown_arity_apply_widget env1 ap1 ret1 exn1 b1)
      as [w1 c1] eqn:K1.
    destruct (make_unknown_arity_apply_widget env ap ret exn b) as [w c] eqn:K0.
    destruct (step_unknown_widget r b1 b c1 c env1 env ap1 ap ret1 ret exn1 exn
                w1 w Hs0 Hc Har Hargs Hret Hbret Hexn Hbexn K1 K0)
      as (r1 & (Hs1 & He1 & Hk1) & Hw & Hbw).
    exists r1. split; [ | split; [ rewrite <- Hb1; exact He1 | rewrite <- Hb1;
      cbn; lia ] ].
    apply sim_add; [ exact Hs1 | ]. intros g [<- | []].
    pose proof (sim_keys _ _ _ _ _ _ Hs1) as Hk.
    split; [ apply fact_ok_here; cbn [rename_fact map_fact]; unfold closure;
             rewrite Hw, renamed_simple by exact Hk; left; reflexivity | ].
    intros n0 [<- | [<- | []]]; [ exact Hbw | unfold closure ].
    destruct (ap_callee ap); cbn; auto.
  - (* known arity, Direct *)
    destruct (Known Known_arity_code_pointer)
      as (r1 & w1 & w & c1 & c & -> & -> & S1 & Hw & Hbw).
    exists r1. destruct S1 as (Hs1 & He1 & Hk1).
    split; [ | split; [ rewrite <- Hb1; exact He1 | rewrite <- Hb1; exact Hk1 ]
      ].
    destruct (env_should_preserve_direct_calls env);
      try (destruct (negb (among id (env_unit_code_ids env))));
      first [ exact Hs1
            | exact (sim_grow _ _ _ _ _ _ _ Hs1 (grows_add_apply _ _ _ _ _ _))
              ].
  - (* known arity, known arity *)
    destruct (Known Known_arity_code_pointer)
      as (r1 & w1 & w & c1 & c & -> & -> & S1 & Hw & Hbw).
    exists r1. destruct S1 as (Hs1 & He1 & Hk1).
    split; [ exact Hs1 | split; [ rewrite <- Hb1; exact He1 | rewrite <- Hb1;
      exact Hk1 ] ].
Qed.

Lemma step_aliases (r : list (node * node)) (a1 a : acc)
    (qs1 qs : list node) (xs1 xs : list simple) :
  sim r a1 a ->
  (∀ q1 s1, (q1, s1) ∈ combine qs1 xs1 ->
     ∃ s, (renamed r q1, s) ∈ combine qs xs ∧ simple_le sol s1 s) ->
  all_below (acc_next a1) qs1 ->
  incl (simple_symbols xs1) (simple_symbols xs) ->
  sim r
    (add_facts (map (fun '(p, s) => Fact_alias p (node_of_simple s))
                  (combine qs1 xs1)) (note_simples xs1 a1))
    (add_facts (map (fun '(p, s) => Fact_alias p (node_of_simple s))
                  (combine qs xs)) (note_simples xs a)).
Proof.
  intros Hs Hpairs Hb Hsym. apply sim_add; [ exact (sim_notes _ _ _ _ _ _ _ _ Hs
    Hsym) | ].
  pose proof (sim_keys _ _ _ _ _ _ Hs) as Hk.
  intros g Hg. apply in_map_iff in Hg as ([q1 s1] & <- & Hin).
  destruct (Hpairs q1 s1 Hin) as (s & Hin' & Hle). split.
  - cbn [rename_fact map_fact]. rewrite renamed_simple by exact Hk.
    destruct Hle as [-> | ((c & ->) & Hu)].
    + apply fact_ok_here. apply in_map_iff. exists (renamed r q1, s).
      split; [ reflexivity | exact Hin' ].
    + eapply fact_ok_pois_here; [ | constructor; exact Hu ].
      apply in_map_iff. exists (renamed r q1, s).
      split; [ reflexivity | exact Hin' ].
  - rewrite acc_next_note_simples. intros n0 [<- | [<- | []]].
    + apply Hb. exact (in_combine_l _ _ _ _ Hin).
    + destruct s1; cbn; auto.
Qed.

Lemma pairs_filter_kept (r : list (node * node)) (mask : list bool)
    (params : list variable) (xs : list simple) (f : simple -> simple) :
  keys_made r -> (∀ s, simple_le sol (f s) s) ->
  ∀ q1 s1,
    (q1, s1) ∈ combine (map Node_variable (filter_kept mask params))
                 (filter_kept mask (map f xs)) ->
    ∃ s, (renamed r q1, s) ∈ combine (map Node_variable params) xs
         ∧ simple_le sol s1 s.
Proof.
  intros Hk Hf q1 s1 H. rewrite <- filter_kept_map in H.
  apply in_combine_filter_kept in H.
  apply in_combine_map_r in H as (s & -> & H). exists s.
  split; [ | apply Hf ].
  apply in_combine_l in H as Hq. apply in_map_iff in Hq as (x & <- & _).
  rewrite renamed_variable' by exact Hk. exact H.
Qed.

Lemma pairs_renamed (r : list (node * node)) (qs1 : list node)
    (xs : list simple) (f : simple -> simple) :
  (∀ s, simple_le sol (f s) s) ->
  ∀ q1 s1,
    (q1, s1) ∈ combine qs1 (map f xs) ->
    ∃ s, (renamed r q1, s) ∈ combine (map (renamed r) qs1) xs
         ∧ simple_le sol s1 s.
Proof.
  intros Hf q1 s1 H. apply in_combine_map_r in H as (s & -> & H). exists s.
  split; [ exact (in_combine_map_l (renamed r) _ _ _ _ H) | apply Hf ].
Qed.

Lemma symbols_map_rewrite (en : rebuild_env) (xs : list simple) :
  incl (simple_symbols (map (rewrite_simple sol en) xs)) (simple_symbols xs).
Proof.
  apply symbols_forall2 with (R := simple_le sol);
    [ apply simple_le_symbols | ].
  induction xs as [ | x xs IH]; constructor;
    [ apply rewrite_simple_le | exact IH ].
Qed.

Lemma filter_kept_incl {A : Type} (mask : list bool) (l : list A) :
  incl (filter_kept mask l) l.
Proof.
  revert mask. induction l as [ | x l IH]; intros mask y Hy; [ destruct Hy | ].
  destruct mask as [ | [] mask]; cbn in Hy.
  - destruct Hy as [<- | Hy]; [ left; reflexivity | right; exact (IH _ _ Hy) ].
  - destruct Hy as [<- | Hy]; [ left; reflexivity | right; exact (IH _ _ Hy) ].
  - right. exact (IH _ _ Hy).
Qed.

Lemma all_below_variables (k : positive) (params : list variable) :
  all_below k (map Node_variable params).
Proof.
  intros n Hn. apply in_map_iff in Hn as (x & <- & _). cbn. exact I.
Qed.

Lemma map_renamed_variables (r : list (node * node)) (params : list variable) :
  keys_made r ->
  map (renamed r) (map Node_variable params) = map Node_variable params.
Proof.
  intros Hk. rewrite map_map. apply map_ext. intros x.
  apply renamed_variable'. exact Hk.
Qed.

Lemma filter_kept_head {A : Type} (mask : list bool) (p : A) (rest : list A) :
  match mask with keep :: _ => keep | [] => true end = true ->
  filter_kept mask (p :: rest) = p :: filter_kept (tl mask) rest.
Proof.
  intros H. destruct mask as [ | [] mask]; cbn; [ | reflexivity | discriminate
    ].
  rewrite filter_kept_nil. reflexivity.
Qed.

Lemma exn_extra_simples_rebuilt (en : rebuild_env) (ap ap1 : apply_expr) :
  ap_exn_extra_args ap1 = rebuild_exn_extra_args sol en ap ->
  exn_extra_simples ap1
  = match lookup (ap_exn_continuation ap) (rconts en) with
    | Some mask =>
      filter_kept (tl mask)
        (map (rewrite_simple sol en) (exn_extra_simples ap))
    | None => map (rewrite_simple sol en) (exn_extra_simples ap)
    end.
Proof.
  intros H. unfold exn_extra_simples. rewrite H. unfold rebuild_exn_extra_args.
  assert (E : map fst (map (fun '(s, kind) => (rewrite_simple sol en s, kind))
                         (ap_exn_extra_args ap))
              = map (rewrite_simple sol en) (map fst (ap_exn_extra_args ap))).
  { rewrite !map_map. apply map_ext. intros [s kind]. reflexivity. }
  destruct (lookup (ap_exn_continuation ap) (rconts en)).
  - rewrite <- filter_kept_map, E. reflexivity.
  - exact E.
Qed.

Lemma step_apply (masks : finite_map (list bool)) (en : rebuild_env)
    (r : list (node * node)) (a1 a : acc) (env1 env : traverse_env)
    (ap ap1 : apply_expr) :
  sim r a1 a -> env_rel masks r (acc_next a1) env1 env -> rconts en = masks ->
  call_conts_kept masks ap = true ->
  ap_callee ap1 = ap_callee ap -> ap_args_arity ap1 = ap_args_arity ap ->
  ap_continuation ap1 = ap_continuation ap ->
  ap_exn_continuation ap1 = ap_exn_continuation ap ->
  ap_return_arity ap1 = ap_return_arity ap ->
  ap_exn_extra_args ap1 = rebuild_exn_extra_args sol en ap ->
  Forall2 arg_le (ap_args ap1) (ap_args ap) ->
  ck_rel (env_should_preserve_direct_calls env) (env_unit_code_ids env)
    (ap_call_kind ap1) (ap_call_kind ap) ->
  ∃ r', stepped r a1 r' (traverse_apply env1 ap1 a1) (traverse_apply env ap a).
Proof.
  intros Hs Henv Hen Hkept Hcal Har Hk Hek Hra Hex Hargs Hck.
  unfold call_conts_kept in Hkept. apply andb_true_iff in Hkept as [Hkr Hke].
  pose proof (exn_extra_simples_rebuilt en ap ap1 Hex) as Hxs.
  rewrite Hen in Hxs.
  unfold traverse_apply. rewrite Hk, Hek, Hra.
  set (k := ap_continuation ap) in *.
  set (ek := ap_exn_continuation ap) in *.
  set (n := length (unarize (ap_return_arity ap))).
  destruct (find_cont env1 k n a1) as [rs1 b1] eqn:K1.
  destruct (find_cont env k n a) as [rs b] eqn:K0.
  destruct (step_find_cont r a1 a b1 b env1 env k (lookup k masks) n n rs1 rs
              Hs (er_conts _ _ _ _ _ Henv k) (fun _ => eq_refl) K1 K0)
    as (r1 & (Hs1 & He1 & Hk1) & Hps1).
  pose proof (sim_keys _ _ _ _ _ _ Hs1) as Hkeys1.
  assert (Hrs : map (renamed r1) rs1 = rs ∧ all_below (acc_next b1) rs1).
  { destruct (lookup k masks) as [mask | ] eqn:Lm; [ | exact Hps1 ].
    destruct Hps1 as (params & -> & ->). rewrite filter_kept_true by exact Hkr.
    split; [ apply map_renamed_variables; exact Hkeys1
           | apply all_below_variables ]. }
  destruct Hrs as [Hrs Hbrs].
  assert (Hcount : lookup ek masks = None ->
                   S (length (ap_exn_extra_args ap1))
                   = S (length (ap_exn_extra_args ap))).
  { intros Hn. rewrite Hex. unfold rebuild_exn_extra_args. rewrite Hen.
    unfold ek in Hn. rewrite Hn.
    rewrite length_map. reflexivity. }
  destruct (find_cont env1 ek (S (length (ap_exn_extra_args ap1))) b1)
    as [eps1 c1] eqn:X1.
  destruct (find_cont env ek (S (length (ap_exn_extra_args ap))) b)
    as [eps c] eqn:X0.
  destruct (step_find_cont r1 b1 b c1 c env1 env ek (lookup ek masks) _ _
              eps1 eps Hs1
              (er_conts _ _ _ _ _ (env_rel_ext _ _ _ _ _ _ _ He1 Hk1 Henv) ek)
              Hcount X1 X0)
    as (r2 & (Hs2 & He2 & Hk2) & Hps2).
  pose proof (sim_keys _ _ _ _ _ _ Hs2) as Hkeys2.
  set (xs := exn_extra_simples ap) in *.
  (* The exception argument. *)
  assert (Exn : ∃ r3 e1 d1 e d,
    match eps1 with
    | [] => fresh c1
    | exn_param :: extra_params =>
      (exn_param,
       add_facts
         (map (fun '(p, s) => Fact_alias p (node_of_simple s))
            (combine extra_params (exn_extra_simples ap1)))
         (note_simples (exn_extra_simples ap1) c1))
    end = (e1, d1)
    ∧ match eps with
      | [] => fresh c
      | exn_param :: extra_params =>
        (exn_param,
         add_facts
           (map (fun '(p, s) => Fact_alias p (node_of_simple s))
              (combine extra_params xs))
           (note_simples xs c))
      end = (e, d)
    ∧ stepped r2 c1 r3 d1 d ∧ renamed r3 e1 = e ∧ below (acc_next d1) e1).
  { destruct (lookup ek masks) as [mask | ] eqn:Lm.
    - destruct Hps2 as (params & -> & ->).
      destruct params as [ | p rest].
      + cbn [map filter_kept]. replace (filter_kept mask (@nil variable))
          with (@nil variable) by (destruct mask; reflexivity).
        cbn [map]. destruct (fresh c1) as [e1 d1] eqn:F1,
          (fresh c) as [e d] eqn:F0.
        destruct (step_fresh r2 c1 c d1 d e1 e Hs2 F1 F0)
          as (r3 & S3 & He & Hbe).
        exists r3, e1, d1, e, d. auto.
      + rewrite (filter_kept_head mask p rest Hke). cbn [map].
        eexists r2, (Node_variable p), _, (Node_variable p), _.
        split; [ reflexivity | split; [ reflexivity | ] ].
        split; [ | split; [ apply renamed_variable'; exact Hkeys2
                          | rewrite acc_next_add_facts,
                              acc_next_note_simples; cbn; exact I ] ].
        apply stepped_same; [ | rewrite acc_next_add_facts,
                                  acc_next_note_simples; lia ].
        rewrite Hxs. apply step_aliases; [ exact Hs2 | | | ].
        * apply pairs_filter_kept; [ exact Hkeys2 | ].
          intros s. apply rewrite_simple_le.
        * apply all_below_variables.
        * eapply incl_tran; [ apply symbols_sublist, filter_kept_incl | ].
          apply symbols_map_rewrite.
    - destruct Hps2 as [Hm Hb2].
      destruct eps1 as [ | p1 rest1]; subst eps; cbn [map].
      + destruct (fresh c1) as [e1 d1] eqn:F1, (fresh c) as [e d] eqn:F0.
        destruct (step_fresh r2 c1 c d1 d e1 e Hs2 F1 F0)
          as (r3 & S3 & He & Hbe).
        exists r3, e1, d1, e, d. auto.
      + eexists r2, p1, _, (renamed r2 p1), _.
        split; [ reflexivity | split; [ reflexivity | ] ].
        split; [ | split; [ reflexivity | ] ].
        * apply stepped_same; [ | rewrite acc_next_add_facts,
                                    acc_next_note_simples; lia ].
          rewrite Hxs. apply step_aliases; [ exact Hs2 | | | ].
          -- apply pairs_renamed. intros s. apply rewrite_simple_le.
          -- intros q Hq. apply Hb2. right. exact Hq.
          -- apply symbols_map_rewrite.
        * rewrite acc_next_add_facts, acc_next_note_simples.
          apply Hb2. left. reflexivity. }
  destruct Exn as (r3 & e1 & d1 & e & d & -> & -> & (Hs3 & He3 & Hk3) & Hre
                   & Hbe).
  assert (Ek : (acc_next a1 <= acc_next c1)%positive) by lia.
  destruct (step_call_kind r3 d1 d env1 env ap1 ap rs1 rs e1 e Hs3
              (er_code _ _ _ _ _ Henv) (er_pres _ _ _ _ _ Henv)
              (er_ucids _ _ _ _ _ Henv) Hcal Har Hargs) as (r4 & S4).
  - rewrite (map_renamed_ext r1 r3 (acc_next b1) rs1); [ exact Hrs | | exact
    Hbrs ].
    exact (ext_trans r1 r2 r3 _ _ Hk2 He2 He3).
  - exact (all_below_mono _ _ _ (Pos.le_trans _ _ _ Hk2 Hk3) Hbrs).
  - exact Hre.
  - exact Hbe.
  - exact Hck.
  - exists r4. eapply stepped_trans; [ | exact S4 ].
    split; [ exact Hs3 | split; [ | lia ] ].
    apply (ext_trans r r1 r3 _ (acc_next b1) Hk1 He1).
    exact (ext_trans r1 r2 r3 _ _ Hk2 He2 He3).
Qed.

Lemma step_fold_subrel {X1 X : Type} (Inv : list (node * node) -> acc -> Prop)
    (R : X1 -> X -> Prop) (f1 : acc -> X1 -> acc) (f : acc -> X -> acc)
    (l1 : list X1) (l : list X) :
  subrel R l1 l ->
  (∀ r a1 r' a1' a', Inv r a1 -> stepped r a1 r' a1' a' -> Inv r' a1') ->
  (∀ x1 x r a1 a, x1 ∈ l1 -> R x1 x -> Inv r a1 -> sim r a1 a ->
     ∃ r', stepped r a1 r' (f1 a1 x1) (f a x)) ->
  (∀ x a', x ∈ l -> grows a' (f a' x)) ->
  ∀ r a1 a, Inv r a1 -> sim r a1 a ->
    ∃ r', stepped r a1 r' (List.fold_left f1 l1 a1) (List.fold_left f l a).
Proof.
  intros H. induction H as [ | l1 l b H IH | x1 x l1 l Hx H IH ];
    intros Hinv Hstep Hgrow r a1 a Hi Hs; cbn [List.fold_left].
  - exists r. apply stepped_refl. exact Hs.
  - apply (IH Hinv Hstep); [ intros y a' Hy; apply Hgrow; right; exact Hy
                           | exact Hi | ].
    apply (sim_grow _ _ _ _ _ _ _ Hs). apply Hgrow. left. reflexivity.
  - destruct (Hstep x1 x r a1 a (or_introl eq_refl) Hx Hi Hs) as (r1 & S1).
    pose proof (Hinv _ _ _ _ _ Hi S1) as Hi1.
    destruct S1 as (Hs1 & He1 & Hk1).
    destruct (IH Hinv
                (fun y1 y r0 b1 b Hy => Hstep y1 y r0 b1 b (or_intror Hy))
                (fun y a' Hy => Hgrow y a' (or_intror Hy)) r1 _ _ Hi1 Hs1)
      as (r2 & (Hs2 & He2 & Hk2)).
    exists r2. split; [ exact Hs2 | split; [ | lia ] ].
    exact (ext_trans _ _ _ _ _ Hk1 He1 He2).
Qed.

Lemma grows_find_cont (env : traverse_env) (c : continuation) (n : nat)
    (a : acc) :
  grows a (snd (find_cont env c n a)).
Proof.
  unfold find_cont. destruct (lookup c (tconts env)); [ apply grows_refl | ].
  destruct (fresh_list n a) as [ps a0] eqn:E. cbn [snd].
  eapply grows_trans; [ | apply grows_add ].
  pose proof (grows_fresh_list n a) as G. rewrite E in G. exact G.
Qed.

Lemma grows_apply_cont_deps (env : traverse_env) (ac : apply_cont) (a : acc) :
  grows a (apply_cont_deps env ac a).
Proof.
  unfold apply_cont_deps.
  destruct (find_cont env (ac_continuation ac) (length (ac_args ac)) a)
    as [ps a0] eqn:E.
  eapply grows_trans; [ | eapply grows_trans; [ apply grows_notes
                                                | apply grows_add ] ].
  pose proof (grows_find_cont env (ac_continuation ac)
                (length (ac_args ac)) a) as G.
  rewrite E in G. exact G.
Qed.

Lemma step_apply_cont (masks : finite_map (list bool)) (en : rebuild_env)
    (r : list (node * node)) (a1 a : acc) (env1 env : traverse_env)
    (ac ac1 : apply_cont) :
  sim r a1 a -> env_rel masks r (acc_next a1) env1 env -> rconts en = masks ->
  rewrite_apply_cont sol en ac = Some ac1 ->
  ∃ r', stepped r a1 r' (apply_cont_deps env1 ac1 a1)
          (apply_cont_deps env ac a).
Proof.
  intros Hs Henv Hen E. unfold rewrite_apply_cont in E.
  destruct (existsb (simple_has_no_source sol) (ac_args ac)); [ discriminate |
    ].
  injection E as <-. rewrite Hen. unfold apply_cont_deps. cbn [ac_args
    ac_continuation].
  set (k := ac_continuation ac). set (xs := ac_args ac).
  set (xs1 := match lookup k masks with
              | Some mask => filter_kept mask (map (rewrite_simple sol en) xs)
              | None => map (rewrite_simple sol en) xs
              end).
  destruct (find_cont env1 k (length xs1) a1) as [ps1 b1] eqn:K1.
  destruct (find_cont env k (length xs) a) as [ps b] eqn:K0.
  assert (Hcount : lookup k masks = None -> length xs1 = length xs).
  { intros Hn. unfold xs1. rewrite Hn. apply length_map. }
  destruct (step_find_cont r a1 a b1 b env1 env k (lookup k masks) _ _ ps1 ps
              Hs (er_conts _ _ _ _ _ Henv k) Hcount K1 K0)
    as (r1 & (Hs1 & He1 & Hk1) & Hps).
  pose proof (sim_keys _ _ _ _ _ _ Hs1) as Hkeys.
  exists r1. split; [ | split; [ exact He1 | rewrite acc_next_add_facts,
                                     acc_next_note_simples; exact Hk1 ] ].
  unfold xs1 in *. destruct (lookup k masks) as [mask | ].
  - destruct Hps as (params & -> & ->). apply step_aliases; [ exact Hs1 | | | ].
    + apply pairs_filter_kept; [ exact Hkeys | intros s; apply rewrite_simple_le
      ].
    + apply all_below_variables.
    + eapply incl_tran; [ apply symbols_sublist, filter_kept_incl | ].
      apply symbols_map_rewrite.
  - destruct Hps as [Hm Hb]. subst ps. apply step_aliases; [ exact Hs1 | | | ].
    + apply pairs_renamed. intros s. apply rewrite_simple_le.
    + exact Hb.
    + apply symbols_map_rewrite.
Qed.


Lemma env_rel_stepped (masks : finite_map (list bool))
    (env1 env : traverse_env) (r : list (node * node)) (a1 : acc)
    (r' : list (node * node)) (a1' a' : acc) :
  env_rel masks r (acc_next a1) env1 env -> stepped r a1 r' a1' a' ->
  env_rel masks r' (acc_next a1') env1 env.
Proof.
  intros H (_ & He & Hk). exact (env_rel_ext _ _ _ _ _ _ _ He Hk H).
Qed.

Lemma step_switch (masks : finite_map (list bool)) (en : rebuild_env)
    (r : list (node * node)) (a1 a : acc) (env1 env : traverse_env)
    (s : simple) (arms : list (Z * apply_cont)) :
  sim r a1 a -> env_rel masks r (acc_next a1) env1 env -> rconts en = masks ->
  ∃ r', stepped r a1 r'
          (traverse_switch env1 (rewrite_simple sol en s)
             (flat_map
                (fun '(d, ac) =>
                   match rewrite_apply_cont sol en ac with
                   | Some ac1 => [(d, ac1)]
                   | None => []
                   end) (Keyed.compact Z.eqb arms)) a1)
          (traverse_switch env s arms a).
Proof.
  intros Hs Henv Hen. unfold traverse_switch.
  pose proof (subrel_flat_map (rewrite_apply_cont sol en)
                (Keyed.compact Z.eqb arms)) as Hsub.
  set (arms1 := flat_map _ (Keyed.compact Z.eqb arms)) in *.
  assert (Hnd : NoDup (map fst arms1)).
  { apply (subrel_nodup_fst _ _ _ (fun p1 p H => proj1 H) Hsub).
    apply (keyed_distinct_NoDup Z.eqb z_eqb_decides_equality).
    exact (keyed_compact_distinct Z Z.eqb z_eqb_decides_equality _ arms). }
  rewrite (keyed_compact_nodup Z.eqb z_eqb_decides_equality arms1 Hnd).
  pose proof (step_cond_any_usage r a1 a env1 env (rewrite_simple sol en s) s
                Hs (er_code _ _ _ _ _ Henv) (rewrite_simple_le sol en s)) as
                  Hs0.
  assert (Hn0 : acc_next (cond_any_usage env1 (rewrite_simple sol en s) a1)
                = acc_next a1).
  { unfold cond_any_usage. destruct (env_current_code_id env1);
      cbn; destruct (rewrite_simple sol en s); reflexivity. }
  destruct (step_fold_subrel
              (fun r0 b1 => env_rel masks r0 (acc_next b1) env1 env)
              _ (fun a0 '(_, ac) => apply_cont_deps env1 ac a0)
              (fun a0 '(_, ac) => apply_cont_deps env ac a0) _ _ Hsub
              (env_rel_stepped masks env1 env)) with (r := r)
    (a1 := cond_any_usage env1 (rewrite_simple sol en s) a1)
    (a := cond_any_usage env s a) as (r' & S).
  - intros [d1 ac1] [d ac] r0 b1 b _ [_ Hac] Hi Hsb. cbn in Hac.
    exact (step_apply_cont masks en r0 b1 b env1 env ac ac1 Hsb Hi Hen Hac).
  - intros [d ac] a' _. apply grows_apply_cont_deps.
  - rewrite Hn0. exact Henv.
  - exact Hs0.
  - exists r'. destruct S as (Hs' & He & Hk). rewrite Hn0 in He, Hk.
    split; [ exact Hs' | split; [ exact He | exact Hk ] ].
Qed.

Lemma rewrite_simple_used (en : rebuild_env) (s : simple) :
  s_has_usage sol (node_of_simple s) = true -> rewrite_simple sol en s = s.
Proof.
  intros H. destruct s as [x | sym | c]; cbn in *; [ | | reflexivity ].
  - unfold is_var_used. destruct (kind_of_variable en x); try reflexivity;
      rewrite H; reflexivity.
  - unfold is_symbol_used. rewrite H. reflexivity.
Qed.

Lemma step_exn_handler (r : list (node * node)) (a1 a : acc)
    (env1 env : traverse_env) (params : list variable)
    (md md1 : handler_metadata) (hb1 hb : expr) :
  sim r a1 a -> env_current_code_id env1 = env_current_code_id env ->
  hm_is_exn_handler md1 = hm_is_exn_handler md ->
  sim r
    (exn_handler_deps env1
       (Handler
          (filter_kept
             (params_to_keep sol (hm_is_exn_handler md)
                (with_kinds params (hm_params_kinds md))) params)
          md1 hb1) a1)
    (exn_handler_deps env (Handler params md hb) a).
Proof.
  intros Hs Hc Hexn. unfold exn_handler_deps. rewrite Hexn.
  destruct params as [ | p rest]; [ cbn; destruct (hm_is_exn_handler md); exact
    Hs | ].
  destruct (hm_is_exn_handler md) eqn:E; [ | destruct (filter_kept _ _); exact
    Hs ].
  replace (filter_kept (params_to_keep sol true
             (with_kinds (p :: rest) (hm_params_kinds md))) (p :: rest))
    with (p :: filter_kept (tl (params_to_keep sol true
             (with_kinds (p :: rest) (hm_params_kinds md)))) rest).
  - apply step_cond_any_usage; [ | exact Hc | left; reflexivity ].
    apply sim_add; [ exact Hs | ]. intros g [<- | []].
    pose proof (sim_keys _ _ _ _ _ _ Hs) as Hk.
    split; [ apply fact_ok_here; cbn; rewrite renamed_variable' by exact Hk;
             left; reflexivity | ].
    intros n0 [<- | []]. exact I.
  - symmetry. apply filter_kept_head.
    destruct (hm_params_kinds md); cbn; rewrite Pos.eqb_refl, orb_true_r;
      reflexivity.
Qed.

(* A field of a block of the first rebuilt unit, at the place of [s]: [s],
   or a constant where [s] or the field has no use. *)
Definition field_le (base : node) (i : nat) (s1 s : simple) : Prop :=
  s1 = s
  ∨ ((∃ c, s1 = Const c)
     ∧ (s_has_usage sol (node_of_simple s) = false
        ∨ s_field_used sol base (Field_block i Value) = false)).

Lemma rebuild_fields_le (en : rebuild_env) (base : node) (fields : list simple)
    (k j : nat) (s1 : simple) :
  (j, s1) ∈ combine (seq k (length (rebuild_fields sol en base k fields)))
               (rebuild_fields sol en base k fields) ->
  ∃ s, (j, s) ∈ combine (seq k (length fields)) fields ∧ field_le base j s1 s.
Proof.
  revert k. induction fields as [ | s fields IH]; intros k H; [ destruct H | ].
  cbn [rebuild_fields length seq combine] in H |- *.
  destruct H as [H | H].
  - injection H as <- <-. exists s. split; [ left; reflexivity | ].
    unfold field_used.
    destruct (s_field_used sol base (Field_block k Value)) eqn:E.
    + destruct (rewrite_simple_le sol en s) as [-> | (Hc & Hu)];
        [ left; reflexivity | right; split; [ exact Hc | left; exact Hu ] ].
    + right. split; [ eexists; reflexivity | right; exact E ].
  - destruct (IH (S k) H) as (s' & Hs' & Hle). exists s'.
    split; [ right; exact Hs' | exact Hle ].
Qed.

Lemma step_block_deps (r : list (node * node)) (a1 a : acc) (en : rebuild_env)
    (base : node) (fields : list simple) :
  sim r a1 a -> (∀ m, base <> Node_made m) ->
  sim r (block_deps base (rebuild_fields sol en base 0 fields) a1)
    (block_deps base fields a).
Proof.
  intros Hs Hb. unfold block_deps.
  pose proof (sim_keys _ _ _ _ _ _ Hs) as Hk.
  assert (Hsym : incl (simple_symbols (rebuild_fields sol en base 0 fields))
                   (simple_symbols fields)).
  { clear. generalize 0. induction fields as [ | s fields IH]; intros k;
      [ apply incl_refl | ].
    cbn [rebuild_fields]. intros x Hx.
    change (s :: fields) with ([s] ++ fields).
    unfold simple_symbols in *. cbn [flat_map] in Hx. rewrite flat_map_app.
    apply in_app_or in Hx as [Hx | Hx]; apply in_or_app.
    - left. unfold field_used in Hx.
      destruct (s_field_used sol base (Field_block k Value));
        [ | destruct Hx ].
      exact (simple_le_symbols _ _ _ (rewrite_simple_le sol en s) x
               ltac:(cbn; rewrite app_nil_r; exact Hx)).
    - right. exact (IH (S k) x Hx). }
  apply sim_add; [ exact (sim_notes _ _ _ _ _ _ _ _ Hs Hsym) | ].
  intros g Hg. apply in_app_or in Hg as [Hg | Hg].
  - apply in_map_iff in Hg as ([i s1] & <- & Hin).
    destruct (rebuild_fields_le en base fields 0 i s1 Hin)
      as (s & Hs' & Hle). split.
    + cbn [rename_fact map_fact]. rewrite renamed_simple by exact Hk.
      rewrite renamed_not_made by assumption.
      destruct Hle as [-> | ((c & ->) & Hu)].
      * apply fact_ok_here. apply in_or_app. left. apply in_map_iff.
        exists (i, s). split; [ reflexivity | exact Hs' ].
      * eapply fact_ok_pois_here; [ | constructor; exact Hu ].
        apply in_or_app. left. apply in_map_iff.
        exists (i, s). split; [ reflexivity | exact Hs' ].
    + intros n0 Hn0. cbn in Hn0. destruct Hn0 as [<- | [<- | []]];
        [ destruct base; cbn; auto; exfalso; exact (Hb _ eq_refl) | ].
      destruct s1; cbn; auto.
  - destruct Hg as [<- | [<- | []]];
      (split; [ apply fact_ok_here; apply in_or_app; right; cbn;
                rewrite renamed_not_made, renamed_consts by assumption;
                auto | ]);
      intros n0 Hn0; cbn in Hn0;
      (destruct Hn0 as [<- | [<- | []]];
       [ destruct base; cbn; auto; exfalso; exact (Hb _ eq_refl)
       | cbn; exact I ]).
Qed.

Lemma free_names_map_rewrite (en : rebuild_env) (args : list simple) :
  incl (free_names_of_simples (map (rewrite_simple sol en) args))
    (free_names_of_simples args).
Proof.
  intros n Hn. unfold free_names_of_simples in *.
  apply in_flat_map in Hn as (s1 & Hs1 & Hn).
  apply in_map_iff in Hs1 as (s & <- & Hs). apply in_flat_map.
  exists s. split; [ exact Hs | ].
  destruct (rewrite_simple_le sol en s) as [E | ((c & E) & _)];
    rewrite E in Hn; [ exact Hn | destruct Hn ].
Qed.

Lemma free_names_not_made (args : list simple) (n : node) :
  n ∈ free_names_of_simples args -> ∀ m, n <> Node_made m.
Proof.
  intros H m ->. unfold free_names_of_simples in H.
  apply in_flat_map in H as (s & _ & H).
  destruct s; cbn in H; intuition discriminate.
Qed.

Lemma step_prim_default (r : list (node * node)) (a1 a : acc)
    (env1 env : traverse_env) (x : variable) (p : prim)
    (args1 args : list simple) :
  sim r a1 a -> env_current_code_id env1 = env_current_code_id env ->
  incl (free_names_of_simples args1) (free_names_of_simples args) ->
  sim r
    (add_facts
       (Fact_any_source (Node_variable x)
        :: map (fun from => Fact_use (Node_variable x) from)
             (free_names_of_simples args1))
       (if prim_has_arbitrary_effects p then cond_any_usage env1 (Var x) a1
        else a1))
    (add_facts
       (Fact_any_source (Node_variable x)
        :: map (fun from => Fact_use (Node_variable x) from)
             (free_names_of_simples args))
       (if prim_has_arbitrary_effects p then cond_any_usage env (Var x) a
        else a)).
Proof.
  intros Hs Hc Hi. pose proof (sim_keys _ _ _ _ _ _ Hs) as Hk.
  assert (Hs0 : sim r
    (if prim_has_arbitrary_effects p then cond_any_usage env1 (Var x) a1
     else a1)
    (if prim_has_arbitrary_effects p then cond_any_usage env (Var x) a
     else a)).
  { destruct (prim_has_arbitrary_effects p); [ | exact Hs ].
    apply step_cond_any_usage; [ exact Hs | exact Hc | left; reflexivity ]. }
  apply sim_add; [ exact Hs0 | ]. intros g [<- | Hg].
  - split; [ apply fact_ok_here; cbn; rewrite renamed_variable' by exact Hk;
             left; reflexivity | intros n0 [<- | []]; exact I ].
  - apply in_map_iff in Hg as (n & <- & Hn).
    pose proof (free_names_not_made _ _ Hn) as Hnm. split.
    + apply fact_ok_here. cbn [rename_fact map_fact].
      rewrite renamed_variable', (renamed_not_made r n) by assumption.
      right. apply in_map. exact (Hi _ Hn).
    + intros n0 [<- | [<- | []]]; [ exact I | ].
      destruct n; cbn; auto. exfalso. exact (Hnm _ eq_refl).
Qed.

Lemma step_accessor (r : list (node * node)) (a1 a : acc) (x : variable)
    (f : field) (s : simple) :
  sim r a1 a ->
  sim r (add_fact (Fact_accessor (Node_variable x) f (node_of_simple s))
           (note_simple s a1))
    (add_fact (Fact_accessor (Node_variable x) f (node_of_simple s))
       (note_simple s a)).
Proof.
  intros Hs. pose proof (sim_keys _ _ _ _ _ _ Hs) as Hk.
  apply sim_add; [ exact (sim_note _ _ _ _ _ _ _ _ Hs (incl_refl _)) | ].
  intros g [<- | []]. split.
  - apply fact_ok_here. cbn. rewrite renamed_variable', renamed_simple
      by exact Hk. left. reflexivity.
  - intros n0 [<- | [<- | []]]; [ exact I | destruct s; cbn; exact I ].
Qed.

Lemma step_any_source_var (r : list (node * node)) (a1 a : acc)
    (x : variable) :
  sim r a1 a ->
  sim r (add_fact (Fact_any_source (Node_variable x)) a1)
    (add_fact (Fact_any_source (Node_variable x)) a).
Proof.
  intros Hs. pose proof (sim_keys _ _ _ _ _ _ Hs) as Hk.
  apply sim_add; [ exact Hs | ]. intros g [<- | []]. split.
  - apply fact_ok_here. cbn. rewrite renamed_variable' by exact Hk.
    left. reflexivity.
  - intros n0 [<- | []]. exact I.
Qed.

Lemma step_box (r : list (node * node)) (a1 a : acc) (x : variable)
    (bn : boxable_number) (s1 s : simple) :
  sim r a1 a -> simple_le sol s1 s ->
  sim r (add_fact (Fact_constructor (Node_variable x) (Field_boxed_number bn)
                     (node_of_simple s1)) (note_simple s1 a1))
    (add_fact (Fact_constructor (Node_variable x) (Field_boxed_number bn)
                 (node_of_simple s)) (note_simple s a)).
Proof.
  intros Hs Hle. pose proof (sim_keys _ _ _ _ _ _ Hs) as Hk.
  apply sim_add;
    [ exact (sim_note _ _ _ _ _ _ _ _ Hs (simple_le_symbols _ _ _ Hle)) | ].
  intros g [<- | []]. split.
  - cbn [rename_fact map_fact]. rewrite renamed_variable', renamed_simple
      by exact Hk.
    destruct Hle as [-> | ((c & ->) & Hu)];
      [ apply fact_ok_here; left; reflexivity | ].
    eapply fact_ok_pois_here; [ left; reflexivity | ].
    constructor. left. exact Hu.
  - intros n0 [<- | [<- | []]]; [ exact I | destruct s1; cbn; exact I ].
Qed.

Lemma step_prim_map (r : list (node * node)) (a1 a : acc)
    (env1 env : traverse_env) (en : rebuild_env) (x : variable) (p : prim)
    (args : list simple) :
  sim r a1 a -> env_current_code_id env1 = env_current_code_id env ->
  (∀ k, p <> Variadic (Make_block k)) ->
  (∀ m t, p <> Heap (Make_block_with_identity m t)) ->
  p <> Region_primitive Local_allocation ->
  (∀ f s, Fact_accessor (Node_variable x) f (node_of_simple s)
            ∈ acc_facts (traverse_prim env x p args a) ->
     s_has_usage sol (node_of_simple s) = true) ->
  sim r (traverse_prim env1 x p (map (rewrite_simple sol en) args) a1)
    (traverse_prim env x p args a).
Proof.
  intros Hs Hc Hmb Hmi Hla Hused.
  assert (Hdef := step_prim_default r a1 a env1 env x p
                    (map (rewrite_simple sol en) args) args Hs Hc
                    (free_names_map_rewrite en args)).
  assert (Use : ∀ f s, Fact_accessor (Node_variable x) f (node_of_simple s)
                         ∈ acc_facts (traverse_prim env x p args a) ->
                       rewrite_simple sol en s = s)
    by (intros f s H; apply rewrite_simple_used; exact (Hused f s H)).
  clear Hused. revert Use. unfold traverse_prim.
  destruct p as [u | b | v | h | rp];
    [ destruct u | | destruct v | destruct h | destruct rp ];
    try (exfalso; first [ exact (Hmb _ eq_refl) | exact (Hmi _ _ eq_refl)
                        | exact (Hla eq_refl) ]);
    try destruct variant_only; try destruct mut;
    destruct args as [ | s0 [ | s1 rest ] ]; cbn [map] in *;
    intros Use; try exact Hdef.
  all: first
    [ apply step_box; [ exact Hs | apply rewrite_simple_le ]
    | rewrite (Use _ s0) by (cbn; intuition);
      first [ apply step_any_source_var; apply step_accessor; exact Hs
            | apply step_accessor; exact Hs ] ].
Qed.

Lemma step_alias_simple (r : list (node * node)) (a1 a : acc) (x : variable)
    (s1 s : simple) :
  sim r a1 a -> simple_le sol s1 s ->
  sim r (add_fact (Fact_alias (Node_variable x) (node_of_simple s1))
           (note_simple s1 a1))
    (add_fact (Fact_alias (Node_variable x) (node_of_simple s))
       (note_simple s a)).
Proof.
  intros Hs Hle. pose proof (sim_keys _ _ _ _ _ _ Hs) as Hk.
  apply sim_add;
    [ exact (sim_note _ _ _ _ _ _ _ _ Hs (simple_le_symbols _ _ _ Hle)) | ].
  intros g [<- | []]. split.
  - cbn [rename_fact map_fact]. rewrite renamed_variable', renamed_simple
      by exact Hk.
    destruct Hle as [-> | ((c & ->) & Hu)];
      [ apply fact_ok_here; left; reflexivity | ].
    eapply fact_ok_pois_here; [ left; reflexivity | constructor; exact Hu ].
  - intros n0 [<- | [<- | []]]; [ exact I | destruct s1; cbn; exact I ].
Qed.

Lemma step_named (r : list (node * node)) (a1 a : acc)
    (env1 env : traverse_env) (en : rebuild_env) (x : variable)
    (kind : flambda_kind) (d : named) :
  sim r a1 a -> env_current_code_id env1 = env_current_code_id env ->
  plain_named kind d = true ->
  (∀ slot decls vs, d <> Closure_of_set slot decls vs) ->
  is_var_used sol x kind = true ->
  (∀ bn args, d = Prim (Unary (Box_number bn)) args ->
     s_field_used sol (Node_variable x) (Field_boxed_number bn) = true) ->
  (∀ f s, Fact_accessor (Node_variable x) f s ∈ acc_facts (traverse_named env x
    d a)
     -> s_has_usage sol (Node_variable x) = true -> s_has_usage sol s = true)
  ->
  sim r (traverse_named env1 x (rebuild_named sol en x d) a1)
    (traverse_named env x d a).
Proof.
  intros Hs Hc Hpl Hnc Hused Hbox Hacc.
  assert (Hx : value_or_naked kind = true ->
               s_has_usage sol (Node_variable x) = true).
  { intros Hv. unfold is_var_used in Hused.
    destruct kind; try discriminate; exact Hused. }
  destruct d as [s | p args | id vs | slot decls vs | w c | pf w c
                | mf mt c ].
  - (* Simple *)
    cbn [rebuild_named traverse_named].
    apply step_alias_simple; [ exact Hs | apply rewrite_simple_le ].
  - (* Prim *)
    destruct (prim_of_block_dec p) as [Hb | Hb].
    + destruct Hb as [(k & ->) | (m & t & ->)]; cbn [rebuild_named
      traverse_named
        traverse_prim]; apply step_block_deps; [ exact Hs | discriminate
                                               | exact Hs | discriminate ].
    + assert (Hla : p <> Region_primitive Local_allocation)
        by (intros ->; discriminate Hpl).
      replace (rebuild_named sol en x (Prim p args))
        with (Prim p (map (rewrite_simple sol en) args)).
      * cbn [traverse_named].
        assert (Gen : (∀ rp, p <> Region_primitive rp) ->
                  sim r (traverse_prim env1 x p (map (rewrite_simple sol en)
                    args)
                           a1) (traverse_prim env x p args a)).
        { intros Hnr. apply step_prim_map; [ exact Hs | exact Hc | | | exact Hla
          | ].
          - intros k0 E. apply Hb. left. exists k0. exact E.
          - intros m t E. apply Hb. right. exists m, t. exact E.
          - intros f s0 Hin. apply (Hacc f _ Hin). apply Hx.
            destruct p; try exact Hpl. exfalso. exact (Hnr _ eq_refl). }
        destruct p as [u | b | v | h | rp];
          try (apply Gen; intros rp0 E; discriminate E).
        destruct rp; try (exfalso; exact (Hla eq_refl));
          unfold traverse_prim; cbv beta iota zeta;
          (apply step_prim_default;
           [ exact Hs | exact Hc | apply free_names_map_rewrite ]).
      * destruct p as [u | | v | h | ]; try reflexivity.
        -- destruct u; try reflexivity. cbn [rebuild_named].
           unfold field_used. rewrite (Hbox _ _ eq_refl). reflexivity.
        -- destruct v; [ exfalso; apply Hb; left; eexists; reflexivity
                       | reflexivity ].
        -- destruct h; try reflexivity.
           exfalso. apply Hb. right. do 2 eexists. reflexivity.
  - discriminate Hpl.
  - exfalso. exact (Hnc slot decls vs eq_refl).
  - discriminate Hpl.
  - (* Project_value_slot_from *)
    cbn [rebuild_named traverse_named].
    rewrite (rewrite_simple_used en c).
    + apply step_accessor. exact Hs.
    + apply (Hacc (Field_value_slot w)); [ cbn; left; reflexivity | ].
      apply Hx. exact Hpl.
  - (* Project_function_slot *)
    cbn [rebuild_named traverse_named].
    rewrite (rewrite_simple_used en c).
    + apply step_accessor. exact Hs.
    + apply (Hacc (Field_function_slot mt)); [ cbn; left; reflexivity | ].
      apply Hx. exact Hpl.
Qed.

Lemma in_compact_lookup {A : Type} (w : positive) (v : A)
    (l : finite_map A) :
  (w, v) ∈ compact l <-> lookup w l = Some v.
Proof. exact (keyed_in_compact Pos.eqb pos_eqb_decides_equality w v l). Qed.

Lemma lookup_rebuild_value_slots (en : rebuild_env) (bound : list node)
    (vs : list (value_slot * simple)) (w : value_slot) :
  lookup w (rebuild_value_slots sol en bound vs)
  = if existsb (fun n => field_used sol n (Field_value_slot w)) bound
    then option_map (rewrite_simple sol en) (lookup w vs) else None.
Proof.
  induction vs as [ | [w' s'] vs IH].
  - cbn. destruct (existsb _ bound); reflexivity.
  - unfold rebuild_value_slots in *. cbn [flat_map].
    destruct (Pos.eqb w w') eqn:E.
    + apply Pos.eqb_eq in E. subst w'.
      destruct (existsb _ bound); cbn; rewrite ?Pos.eqb_refl; try reflexivity.
      exact IH.
    + destruct (existsb (fun n => field_used sol n (Field_value_slot w'))
                  bound); cbn; rewrite ?E; exact IH.
Qed.

Lemma step_closure (r : list (node * node)) (a1 a : acc)
    (env1 env : traverse_env) (en : rebuild_env) (x : variable)
    (slot f : function_slot) (id : code_id) (vs : list (value_slot * simple))
    (bound : list node) :
  sim r a1 a ->
  incl (env_value_slots_to_keep env1) (env_value_slots_to_keep env) ->
  code_ok cds1 cds id ->
  sim r (traverse_named env1 x
           (rebuild_set sol en bound (Closure_of_set slot [(f, id)] vs)) a1)
    (traverse_named env x (Closure_of_set slot [(f, id)] vs) a).
Proof.
  intros Hs Hslots Hok. cbn [rebuild_set traverse_named].
  unfold traverse_closure_of_set.
  assert (C : compact [(f, id)] = [(f, id)]).
  { unfold compact, tabulate, domain. cbn. rewrite Pos.eqb_refl.
    reflexivity. }
  rewrite C. cbn [map fst among existsb]. rewrite orb_false_r.
  destruct (Pos.eqb slot f) eqn:E; [ | exact Hs ].
  apply Pos.eqb_eq in E. subst f. cbn [closures_of_set].
  rewrite Pos.eqb_refl. cbn [map].
  unfold record_set_of_closures_deps. cbn [List.fold_left].
  pose proof (sim_keys _ _ _ _ _ _ Hs) as Hk.
  set (vs1 := rebuild_value_slots sol en bound vs).
  set (k1 := filter (fun '(w, _) => should_keep_value_slot env1 w)
               (compact vs1)).
  set (k0 := filter (fun '(w, _) => should_keep_value_slot env w)
               (compact vs)).
  assert (Hrel : ∀ w s1, (w, s1) ∈ k1 ->
                 ∃ s, (w, s) ∈ k0 ∧ simple_le sol s1 s).
  { intros w s1 H. unfold k1 in H. apply filter_In in H as [H Hkeep].
    apply in_compact_lookup in H. unfold vs1 in H.
    rewrite lookup_rebuild_value_slots in H.
    destruct (existsb _ bound); [ | discriminate ].
    destruct (lookup w vs) as [s | ] eqn:L; [ | discriminate ].
    injection H as <-. exists s. split; [ | apply rewrite_simple_le ].
    unfold k0. apply filter_In. split; [ apply in_compact_lookup; exact L | ].
    unfold should_keep_value_slot in *. apply FlambdaMapProofs.among_In in
      Hkeep. apply FlambdaMapProofs.among_In. exact (Hslots _ Hkeep). }
  assert (Er : rename_cld r (Closure_dep (Node_variable x) id)
               = Closure_dep (Node_variable x) id)
    by (unfold rename_cld; cbn; rewrite renamed_variable' by exact Hk;
        reflexivity).
  assert (Hs1 := sim_push_closure _ _ _ r a1 a (Closure_dep (Node_variable x)
    id)
                   Hs I Hok).
  rewrite Er in Hs1.
  apply sim_add.
  - apply sim_notes; [ exact Hs1 | ].
    intros sym Hsym. unfold simple_symbols in *.
    apply in_flat_map in Hsym as (s1 & Hs1' & Hsym).
    apply in_map_iff in Hs1' as ([w s1'] & E & Hin). cbn in E. subst s1'.
    destruct (Hrel w s1 Hin) as (s & Hin0 & Hle).
    apply in_flat_map. exists s. split.
    + apply in_map_iff. exists (w, s). split; [ reflexivity | exact Hin0 ].
    + destruct Hle as [-> | ((c & ->) & _)]; [ exact Hsym | destruct Hsym ].
  - intros g Hg. rewrite acc_next_note_simples. cbn [acc_next push_closure_dep].
    apply in_app_or in Hg as [Hg | Hg].
    + apply in_map_iff in Hg as ([w s1] & <- & Hin).
      destruct (Hrel w s1 Hin) as (s & Hin0 & Hle). split.
      * cbn [rename_fact map_fact]. rewrite renamed_variable', renamed_simple
          by exact Hk.
        destruct Hle as [-> | ((c & ->) & Hu)].
        -- apply fact_ok_here. apply in_or_app. left. apply in_map_iff.
           exists (w, s). split; [ reflexivity | exact Hin0 ].
        -- eapply fact_ok_pois_here; [ | constructor; left; exact Hu ].
           apply in_or_app. left. apply in_map_iff.
           exists (w, s). split; [ reflexivity | exact Hin0 ].
      * intros n0 [<- | [<- | []]]; [ exact I | destruct s1; cbn; exact I ].
    + destruct Hg as [<- | []]. split.
      * apply fact_ok_here. apply in_or_app. right. cbn.
        rewrite renamed_variable' by exact Hk. left. reflexivity.
      * intros n0 [<- | [<- | []]]; exact I.
Qed.

End Steps.

(* ======================================================================== *)
(* The run of the unit alone, through terms                                 *)
(* ======================================================================== *)

Ltac grow_fresh b :=
  let x := fresh "x" in let b' := fresh "a" in let E := fresh "E" in
  destruct (FlambdaReaperTraversal.fresh b) as [x b'] eqn:E;
  eapply grows_trans;
    [ let G := fresh "G" in
      pose proof (grows_fresh b) as G; rewrite E in G; exact G | ].

Lemma grows_cond_any_usage (env : traverse_env) (s : simple) (a : acc) :
  grows a (cond_any_usage env s a).
Proof.
  unfold cond_any_usage. eapply grows_trans; [ apply grows_note | ].
  destruct (env_current_code_id env); apply grows_add.
Qed.

Lemma grows_cond_alias (env : traverse_env) (x y : node) (a : acc) :
  grows a (cond_alias env x y a).
Proof. unfold cond_alias. destruct (env_current_code_id env); apply grows_add.
Qed.

Lemma grows_block_deps (base : node) (fields : list simple) (a : acc) :
  grows a (block_deps base fields a).
Proof.
  unfold block_deps. eapply grows_trans; [ apply grows_notes | apply grows_add
    ].
Qed.

Lemma grows_traverse_prim (env : traverse_env) (x : variable) (p : prim)
    (args : list simple) (a : acc) :
  grows a (traverse_prim env x p args a).
Proof.
  unfold traverse_prim.
  assert (Gacc : ∀ f s a0, grows a0
     (add_fact (Fact_accessor (Node_variable x) f (node_of_simple s))
        (note_simple s a0)))
    by (intros; eapply grows_trans; [ apply grows_note | apply grows_add ]).
  assert (Gdef : grows a
    (add_facts
       (Fact_any_source (Node_variable x)
        :: map (fun from => Fact_use (Node_variable x) from)
             (free_names_of_simples args))
       (if prim_has_arbitrary_effects p then cond_any_usage env (Var x) a
        else a))).
  { eapply grows_trans; [ | apply grows_add ].
    destruct (prim_has_arbitrary_effects p);
      [ apply grows_cond_any_usage | apply grows_refl ]. }
  destruct p as [u | b | v | h | rp];
    [ destruct u | | destruct v | destruct h | destruct rp ];
    try destruct variant_only; try destruct mut;
    destruct args as [ | s0 [ | s1 rest ] ]; try exact Gdef;
    try apply grows_block_deps; try apply grows_refl; try apply Gacc;
    try (eapply grows_trans; [ apply Gacc | apply grows_add ]);
    (eapply grows_trans; [ apply grows_note | apply grows_add ]).
Qed.

Lemma grows_record_set (env : traverse_env)
    (closures : list (node * option code_id))
    (function_slots : list (function_slot * node))
    (value_slots : list (value_slot * simple)) (a : acc) :
  grows a (record_set_of_closures_deps env closures function_slots
             value_slots a).
Proof.
  unfold record_set_of_closures_deps. apply grows_fold.
  intros a' [base code] _. eapply grows_trans; [ | eapply grows_trans;
    [ apply grows_notes | apply grows_add ] ].
  destruct code; [ apply grows_push_closure | apply grows_refl ].
Qed.

Lemma grows_closures_of_set (x : variable) (slot : function_slot)
    (others : list (function_slot * variable))
    (decls : list (function_slot * code_id)) (a : acc) :
  grows a (snd (closures_of_set x slot others decls a)).
Proof.
  revert a. induction decls as [ | [f id] decls IH]; intros a;
    [ apply grows_refl | ].
  cbn [closures_of_set].
  destruct (if Pos.eqb f slot then (Node_variable x, a)
            else match lookup f others with
                 | Some y => (Node_variable y, a)
                 | None => fresh a
                 end) as [name a0] eqn:E.
  destruct (closures_of_set x slot others decls a0) as [rest a2] eqn:E2.
  cbn [snd]. eapply grows_trans; [ | specialize (IH a0); rewrite E2 in IH;
                                     exact IH ].
  destruct (Pos.eqb f slot); [ injection E as _ <-; apply grows_refl | ].
  destruct (lookup f others); [ injection E as _ <-; apply grows_refl | ].
  pose proof (grows_fresh a) as G. rewrite E in G. exact G.
Qed.

Lemma grows_traverse_named (env : traverse_env) (x : variable) (d : named)
    (a : acc) :
  grows a (traverse_named env x d a).
Proof.
  destruct d as [s | p args | id vs | slot decls vs | w c | pf w c
                | mf mt c ]; cbn [traverse_named].
  - eapply grows_trans; [ apply grows_note | apply grows_add ].
  - apply grows_traverse_prim.
  - apply grows_record_set.
  - unfold traverse_closure_of_set.
    destruct (among slot (map fst (compact decls))); [ | apply grows_refl ].
    destruct (closures_of_set x slot (set_closures_of env x) (compact decls) a)
      as [cl a0] eqn:E.
    eapply grows_trans; [ | apply grows_record_set ].
    pose proof (grows_closures_of_set x slot (set_closures_of env x)
                  (compact decls) a) as G. rewrite E in G. exact G.
  - eapply grows_trans; [ apply grows_note | apply grows_add ].
  - eapply grows_trans; [ apply grows_note | apply grows_add ].
  - eapply grows_trans; [ apply grows_note | apply grows_add ].
Qed.

Lemma grows_known_widget (env : traverse_env) (ap : apply_expr)
    (ret : list node) (exn : node) (a : acc) :
  grows a (snd (make_known_arity_apply_widget env ap ret exn a)).
Proof.
  unfold make_known_arity_apply_widget.
  do 3 (match goal with
        | |- context [FlambdaReaperTraversal.fresh ?b] => grow_fresh b
        end). cbn [snd].
  eapply grows_trans; [ apply grows_notes | ].
  eapply grows_trans; [ apply grows_add | apply grows_cond_alias ].
Qed.

Lemma grows_unknown_deps (ret : list node) (exn cal : node)
    (gw : list (list simple * node)) (a : acc) :
  grows a (unknown_arity_apply_deps ret exn cal gw a).
Proof.
  revert a. induction gw as [ | [g w] rest IH]; intros a;
    [ apply grows_refl | ].
  cbn [unknown_arity_apply_deps]. eapply grows_trans; [ apply grows_add | ].
  destruct rest as [ | [g' w'] rest']; [ apply grows_add | ].
  grow_fresh (add_facts
      (map (λ '(i, s), Fact_argument (node_of_simple s) i w) (indexed g)
       ++ [Fact_accessor exn (Field_return_of_call Exn) w;
           Fact_accessor cal Field_code_id_of_call_witness w]) a).
  eapply grows_trans; [ apply grows_add | apply IH ].
Qed.

Lemma grows_unknown_widget (env : traverse_env) (ap : apply_expr)
    (ret : list node) (exn : node) (a : acc) :
  grows a (snd (make_unknown_arity_apply_widget env ap ret exn a)).
Proof.
  unfold make_unknown_arity_apply_widget.
  grow_fresh a. eapply grows_trans; [ apply grows_add | ].
  set (g := at_least_one_group _).
  destruct (fresh_list (length g) (add_fact (Fact_any_usage x) a0))
    as [ws a2] eqn:E2.
  eapply grows_trans;
    [ pose proof (grows_fresh_list (length g)
                    (add_fact (Fact_any_usage x) a0)) as G;
      rewrite E2 in G; exact G | ].
  eapply grows_trans; [ apply grows_notes | ].
  eapply grows_trans; [ apply grows_unknown_deps | ].
  grow_fresh (unknown_arity_apply_deps ret exn x (combine g ws)
                (note_simples (ap_args ap) a2)).
  destruct ws; cbn [snd]; [ apply grows_refl | apply grows_cond_alias ].
Qed.

Lemma grows_call_kind (env : traverse_env) (ap : apply_expr) (exn : node)
    (ret : list node) (a : acc) :
  grows a (traverse_call_kind env ap exn ret a).
Proof.
  unfold traverse_call_kind. eapply grows_trans; [ apply grows_note | ].
  destruct (ap_call_kind ap) as [id | | ids].
  - destruct (make_known_arity_apply_widget env ap ret exn
                (note_simple (ap_callee ap) a)) as [w a0] eqn:E.
    eapply grows_trans;
      [ pose proof (grows_known_widget env ap ret exn
                      (note_simple (ap_callee ap) a)) as G;
        rewrite E in G; exact G | ].
    eapply grows_trans; [ apply grows_add | ].
    destruct (env_should_preserve_direct_calls env);
      try destruct (negb (among id (env_unit_code_ids env)));
      first [ apply grows_add_apply | apply grows_refl ].
  - destruct (make_unknown_arity_apply_widget env ap ret exn
                (note_simple (ap_callee ap) a)) as [w a0] eqn:E.
    eapply grows_trans;
      [ pose proof (grows_unknown_widget env ap ret exn
                      (note_simple (ap_callee ap) a)) as G;
        rewrite E in G; exact G | apply grows_add ].
  - destruct (make_known_arity_apply_widget env ap ret exn
                (note_simple (ap_callee ap) a)) as [w a0] eqn:E.
    eapply grows_trans;
      [ pose proof (grows_known_widget env ap ret exn
                      (note_simple (ap_callee ap) a)) as G;
        rewrite E in G; exact G | apply grows_add ].
Qed.

Lemma grows_traverse_apply (env : traverse_env) (ap : apply_expr) (a : acc) :
  grows a (traverse_apply env ap a).
Proof.
  unfold traverse_apply.
  destruct (find_cont env (ap_continuation ap)
              (length (unarize (ap_return_arity ap))) a) as [rs a0] eqn:E0.
  eapply grows_trans;
    [ pose proof (grows_find_cont env (ap_continuation ap)
                    (length (unarize (ap_return_arity ap))) a) as G;
      rewrite E0 in G; exact G | ].
  destruct (find_cont env (ap_exn_continuation ap)
              (S (length (ap_exn_extra_args ap))) a0) as [eps a2] eqn:E2.
  eapply grows_trans;
    [ pose proof (grows_find_cont env (ap_exn_continuation ap)
                    (S (length (ap_exn_extra_args ap))) a0) as G;
      rewrite E2 in G; exact G | ].
  destruct eps as [ | p rest].
  - grow_fresh a2. apply grows_call_kind.
  - eapply grows_trans; [ | apply grows_call_kind ].
    eapply grows_trans; [ apply grows_notes | apply grows_add ].
Qed.

Lemma grows_traverse_switch (env : traverse_env) (s : simple)
    (arms : list (Z * apply_cont)) (a : acc) :
  grows a (traverse_switch env s arms a).
Proof.
  unfold traverse_switch. eapply grows_trans; [ apply grows_cond_any_usage | ].
  apply grows_fold. intros a' [d ac] _. apply grows_apply_cont_deps.
Qed.

Lemma grows_exn_handler (env : traverse_env) (h : continuation_handler)
    (a : acc) :
  grows a (exn_handler_deps env h a).
Proof.
  unfold exn_handler_deps. destruct h as [[ | p rest] md hb];
    [ apply grows_refl | ].
  destruct (hm_is_exn_handler md); [ | apply grows_refl ].
  eapply grows_trans; [ apply grows_add | apply grows_cond_any_usage ].
Qed.

Lemma grows_traverse :
  (∀ e, plain_expr e = true -> ∀ env a, grows a (traverse env e a))
  ∧ (∀ h, plain_handler h = true ->
       ∀ env a, grows a (traverse_handler env h a)).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [plain_expr plain_handler traverse traverse_handler].
  - intros x kind d body IH H env a. apply andb_true_iff in H as [_ H].
    eapply grows_trans; [ apply grows_traverse_named | exact (IH H env _) ].
  - intros bindings body _ _ H. discriminate H.
  - intros k h body IHh IH H env a. apply andb_true_iff in H as [Hh Hb].
    eapply grows_trans; [ exact (IHh Hh env a) | ].
    eapply grows_trans; [ apply grows_exn_handler | exact (IH Hb _ _) ].
  - intros handlers body IHh IH H env a. apply andb_true_iff in H as [Hh Hb].
    eapply grows_trans; [ | exact (IH Hb _ _) ].
    apply grows_fold. intros a' [k h] Hin.
    apply (IHh k h Hin). rewrite forallb_forall in Hh.
    exact (Hh (k, h) Hin).
  - intros ap _ env a. apply grows_traverse_apply.
  - intros ac _ env a. apply grows_apply_cont_deps.
  - intros s arms _ env a. apply grows_traverse_switch.
  - intros _ env a. apply grows_refl.
  - intros params md handler IH H env a. exact (IH H env a).
Qed.
