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


(* Proofs of the statements of FlambdaJoinTypesSpec.v. Nothing here
   needs review.

   The functions that call the join of types are proved sound once,
   for a list of [Simple]s with kinds at which the things given and
   the answers are kinded ([kinds_with]): the aliases of a type, to
   which are added, or not, the names that have equations inside the
   type, and, or not, a [Simple] of no kind for each test of
   well-formedness that the type fails. The aliases alone give the
   contract of the join of types ([n_way_join_sound]). With the
   names, at the environments that have the names of the equations
   inside their types at their kinds, they give the contract for
   those names ([n_way_join_equations_kinded]). With the checks, at
   the environments whose types are well formed, they give the
   well-formedness of the type answered ([n_way_join_well_formed]). *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeMeaningBoolSpec FlambdaTypeMeaningBoolProofs
  FlambdaTypeWellFormedSpec FlambdaTypeWellFormedProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvKindedSpec FlambdaJoinEnv FlambdaJoinTypes FlambdaJoinSpec
  FlambdaJoinProofs FlambdaJoinTypesSpec.
Import ListNotations.

(* ======================================================================== *)
(* Sets of numbers, and lists of contributors                               *)
(* ======================================================================== *)

Lemma numbers_union_in (n : Z) (numbers numbers' : list Z) :
  n ∈ numbers_union numbers numbers' <-> n ∈ numbers ∨ n ∈ numbers'.
Proof.
  unfold numbers_union.
  rewrite <- (keyed_among_in Z Z.eqb Z.eqb_eq n (Keyed.dedup Z.eqb _)).
  rewrite (keyed_among_dedup Z Z.eqb Z.eqb_eq).
  rewrite (keyed_among_in Z Z.eqb Z.eqb_eq).
  apply in_app_iff.
Qed.

(* What [known_contents] answers: the contents of the list, before what
   it was given to start with. *)
Lemma known_contents_spec {A : Type}
    (or_unknowns : list (Index.t * Or_unknown.t A)) :
  ∀ (acc contents : list (Index.t * A)),
    known_contents acc or_unknowns = Some contents ->
    map fst contents = rev (map fst or_unknowns) ++ map fst acc
    ∧ (∀ (index : Index.t) (a : A),
         In (index, a) contents ->
         In (index, a) acc ∨ In (index, Or_unknown.Known a) or_unknowns)
    ∧ (∀ (index : Index.t) (a : Or_unknown.t A),
         In (index, a) or_unknowns ->
         ∃ a' : A, a = Or_unknown.Known a' ∧ In (index, a') contents)
    ∧ ∀ (index : Index.t) (a : A), In (index, a) acc -> In (index, a) contents.
Proof.
  induction or_unknowns as [ | [id [a | ]] or_unknowns IH];
    intros acc contents Hcontents; simpl in Hcontents.
  - injection Hcontents as <-. simpl. repeat split; auto.
    intros index a [].
  - destruct (IH _ _ Hcontents) as (Hfst & Hfrom & Hto & Hacc).
    repeat split.
    + rewrite Hfst. simpl. rewrite <- app_assoc. reflexivity.
    + intros index a' Hin.
      destruct (Hfrom index a' Hin) as [[Heq | Hin'] | Hin'].
      * right. left. congruence.
      * left. exact Hin'.
      * right. right. exact Hin'.
    + intros index a' [Heq | Hin].
      * injection Heq as <- <-. exists a. split; [ reflexivity | ].
        apply Hacc. left. reflexivity.
      * exact (Hto index a' Hin).
    + intros index a' Hin. apply Hacc. right. exact Hin.
  - discriminate.
Qed.

(* A list made of another element by element, each with its index, or
   nothing if an element gives nothing: what the functions of the
   traversal that gather a list with an exception compute. *)
Fixpoint indexed_map_opt {A B : Type} (f : Index.t -> A -> option B)
    (l : list (Index.t * A)) : option (list (Index.t * B)) :=
  match l with
  | [] => Some []
  | (id, a) :: l =>
    match f id a, indexed_map_opt f l with
    | Some b, Some l' => Some ((id, b) :: l')
    | _, _ => None
    end
  end.

Lemma indexed_map_opt_spec {A B : Type} (f : Index.t -> A -> option B)
    (l : list (Index.t * A)) :
  ∀ l' : list (Index.t * B),
    indexed_map_opt f l = Some l' ->
    map fst l' = map fst l
    ∧ (∀ (index : Index.t) (a : A),
         In (index, a) l -> ∃ b : B, f index a = Some b ∧ In (index, b) l')
    ∧ ∀ (index : Index.t) (b : B),
        In (index, b) l' -> ∃ a : A, f index a = Some b ∧ In (index, a) l.
Proof.
  induction l as [ | [id a] l IH]; intros l' Hl'; simpl in Hl'.
  - injection Hl' as <-. split; [ reflexivity | ].
    split; [ intros ? ? [] | intros ? ? [] ].
  - destruct (f id a) as [b | ] eqn:Hb; [ | discriminate ].
    destruct (indexed_map_opt f l) as [rest | ]; [ | discriminate ].
    injection Hl' as <-.
    destruct (IH rest eq_refl) as (Hfst & Hto & Hfrom).
    split; [ simpl; rewrite Hfst; reflexivity | ].
    split.
    + intros index a' [[= <- <-] | Hin].
      * exists b. split; [ exact Hb | left; reflexivity ].
      * destruct (Hto index a' Hin) as (b' & Hb' & Hin').
        exists b'. split; [ exact Hb' | right; exact Hin' ].
    + intros index b' [[= <- <-] | Hin].
      * exists a. split; [ exact Hb | left; reflexivity ].
      * destruct (Hfrom index b' Hin) as (a' & Ha' & Hin').
        exists a'. split; [ exact Ha' | right; exact Hin' ].
Qed.

(* The lists that the traversal gathers, as such lists. *)
Lemma all_known_eq {A : Type} (l : list (Index.t * Or_unknown.t A)) :
  all_known l
  = indexed_map_opt
      (fun _ a =>
         match a with
         | Or_unknown.Known a => Some a
         | Or_unknown.Unknown => None
         end)
      l.
Proof.
  induction l as [ | [id [a | ]] l IH]; simpl; [ reflexivity | | reflexivity ].
  rewrite IH. destruct (indexed_map_opt _ l); reflexivity.
Qed.

Lemma alias_simples_eq (ts : list (Index.t * flambda_type)) :
  alias_simples ts = indexed_map_opt (fun _ ty => get_alias_opt ty) ts.
Proof.
  induction ts as [ | [id ty] ts IH]; simpl; [ reflexivity | ].
  rewrite IH. destruct (get_alias_opt ty); reflexivity.
Qed.

Lemma fields_at_eq (index : nat) (fields : list (Index.t * list flambda_type)) :
  fields_at index fields
  = indexed_map_opt (fun _ fields_of_id => nth_error fields_of_id index)
      fields.
Proof.
  induction fields as [ | [id fields_of_id] fields IH]; simpl;
    [ reflexivity | ].
  rewrite IH. destruct (nth_error fields_of_id index); reflexivity.
Qed.

(* ======================================================================== *)
(* The fold over a map that threads a state                                 *)
(* ======================================================================== *)

Section Fold.

Context {K : Type} (eqb : K -> K -> bool) (Heqb : decides_equality eqb).
Context {S : Type} (P : S -> Prop) (R : S -> S -> Prop).

Hypothesis R_refl : ∀ s : S, R s s.
Hypothesis R_trans : ∀ s1 s2 s3 : S, R s1 s2 -> R s2 s3 -> R s1 s3.

Lemma among_cons_other (key first : K) (keys : list K) :
  Keyed.among eqb first keys = false ->
  Keyed.among eqb key keys = true ->
  eqb key first = false.
Proof.
  intros Hfirst Hkey.
  destruct (keyed_eqb_spec eqb Heqb key first) as [-> | _];
    [ congruence | reflexivity ].
Qed.

(* What the fold answers, if each step keeps [P] and takes the state
   to one that [R] relates to it: the map answered has no key twice;
   what it binds a key to was answered by the step of that key, at a
   state between the first and the last; and a key that it does not
   bind had a step that answered nothing. *)
Lemma tabulate_fold_opt_spec {A : Type} (f : K -> S -> option (option A * S))
    (keys : list K) :
  Keyed.distinct eqb keys = true ->
  (∀ (key : K) (s : S) (a : option A) (s' : S),
     Keyed.among eqb key keys = true ->
     P s -> f key s = Some (a, s') -> P s' ∧ R s s') ->
  ∀ (s : S) (out : Keyed.finite_map K A) (s' : S),
    P s ->
    Keyed.tabulate_fold_opt f keys s = Some (out, s') ->
    P s' ∧ R s s'
    ∧ (∀ (key : K) (a : A),
         In (key, a) out -> Keyed.lookup eqb key out = Some a)
    ∧ (∀ (key : K) (a : A),
         Keyed.lookup eqb key out = Some a ->
         Keyed.among eqb key keys = true
         ∧ ∃ s1 s2 : S,
             P s1 ∧ R s s1 ∧ f key s1 = Some (Some a, s2) ∧ R s2 s')
    ∧ ∀ key : K,
        Keyed.among eqb key keys = true ->
        (∃ a : A, Keyed.lookup eqb key out = Some a)
        ∨ ∃ s1 s2 : S, P s1 ∧ R s s1 ∧ f key s1 = Some (None, s2).
Proof.
  induction keys as [ | first keys IH]; intros Hdistinct Hstep s out s' HP
    Hfold; simpl in Hfold.
  - injection Hfold as <- <-.
    split; [ exact HP | ]. split; [ apply R_refl | ].
    split; [ intros ? ? [] | ].
    split; [ intros ? ? [=] | intros ? [=] ].
  - simpl in Hdistinct. apply andb_true_iff in Hdistinct as [Hfirst Hdistinct].
    apply negb_true_iff in Hfirst.
    destruct (f first s) as [[a s1] | ] eqn:Hf; [ | discriminate ].
    destruct (Keyed.tabulate_fold_opt f keys s1) as [[rest s2] | ]
      eqn:Hrest; [ | discriminate ].
    injection Hfold as <- <-.
    assert (Hfirst_among : Keyed.among eqb first (first :: keys) = true).
    { rewrite keyed_among_cons, (keyed_eqb_refl eqb Heqb). reflexivity. }
    destruct (Hstep first s a s1 Hfirst_among HP Hf) as (HP1 & HR1).
    assert (Hstep' :
      ∀ (key : K) (s : S) (a : option A) (s' : S),
        Keyed.among eqb key keys = true ->
        P s -> f key s = Some (a, s') -> P s' ∧ R s s').
    { intros key s0 a0 s0' Hkey. apply Hstep.
      rewrite keyed_among_cons, Hkey. apply orb_true_r. }
    destruct (IH Hdistinct Hstep' s1 rest s2 HP1 Hrest)
      as (HP2 & HR2 & Hin & Hfound & Hall).
    split; [ exact HP2 | ]. split; [ exact (R_trans _ _ _ HR1 HR2) | ].
    (* A key of the rest is not the first. *)
    assert (Hother :
      ∀ (key : K) (b : A),
        Keyed.lookup eqb key rest = Some b -> eqb key first = false).
    { intros key b Hkey. destruct (Hfound key b Hkey) as (Hamong & _).
      exact (among_cons_other key first keys Hfirst Hamong). }
    split; [ | split ].
    + intros key b Hb. destruct a as [a | ].
      * destruct Hb as [[= <- <-] | Hb].
        -- apply (keyed_lookup_head eqb Heqb).
        -- pose proof (Hin key b Hb) as Hkey. simpl.
           rewrite (Hother key b Hkey). exact Hkey.
      * exact (Hin key b Hb).
    + intros key b Hkey.
      assert (Hrest_case :
        Keyed.lookup eqb key rest = Some b ->
        Keyed.among eqb key (first :: keys) = true
        ∧ ∃ s1' s2' : S,
            P s1' ∧ R s s1' ∧ f key s1' = Some (Some b, s2') ∧ R s2' s2).
      { intros Hkey'.
        destruct (Hfound key b Hkey') as (Hamong & s1' & s2' & HP' & HR' & Hf'
                                          & HR'').
        split; [ rewrite keyed_among_cons, Hamong; apply orb_true_r | ].
        exists s1', s2'. split; [ exact HP' | ].
        split; [ exact (R_trans _ _ _ HR1 HR') | ].
        split; [ exact Hf' | exact HR'' ]. }
      destruct a as [a | ]; [ | exact (Hrest_case Hkey) ].
      simpl in Hkey.
      destruct (keyed_eqb_spec eqb Heqb key first) as [-> | _];
        [ | exact (Hrest_case Hkey) ].
      injection Hkey as <-.
      split; [ exact Hfirst_among | ].
      exists s, s1. split; [ exact HP | ]. split; [ apply R_refl | ].
      split; [ exact Hf | exact HR2 ].
    + intros key Hkey. rewrite keyed_among_cons in Hkey.
      destruct (keyed_eqb_spec eqb Heqb key first) as [Heq | Hne].
      * subst key. destruct a as [a | ].
        -- left. exists a. apply (keyed_lookup_head eqb Heqb).
        -- right. exists s, s1. split; [ exact HP | ].
           split; [ apply R_refl | exact Hf ].
      * simpl in Hkey. destruct (Hall key Hkey) as [[b Hb] | Hnone].
        -- left. exists b. destruct a as [a | ]; [ | exact Hb ].
           simpl. rewrite (Hother key b Hb). exact Hb.
        -- right. destruct Hnone as (s1' & s2' & HP' & HR' & Hf').
           exists s1', s2'. split; [ exact HP' | ].
           split; [ exact (R_trans _ _ _ HR1 HR') | exact Hf' ].
Qed.

(* The same of the fold over the bindings of a map. *)
Lemma filter_map_fold_spec {A B : Type}
    (g : K -> B -> S -> option (option A * S)) (m : Keyed.finite_map K B) :
  (∀ (key : K) (b : B) (s : S) (a : option A) (s' : S),
     Keyed.lookup eqb key m = Some b ->
     P s -> g key b s = Some (a, s') -> P s' ∧ R s s') ->
  ∀ (s : S) (out : Keyed.finite_map K A) (s' : S),
    P s ->
    Keyed.filter_map_fold eqb g m s = Some (out, s') ->
    P s' ∧ R s s'
    ∧ (∀ (key : K) (a : A),
         In (key, a) out -> Keyed.lookup eqb key out = Some a)
    ∧ (∀ (key : K) (a : A),
         Keyed.lookup eqb key out = Some a ->
         ∃ (b : B) (s1 s2 : S),
           Keyed.lookup eqb key m = Some b
           ∧ P s1 ∧ R s s1 ∧ g key b s1 = Some (Some a, s2) ∧ R s2 s')
    ∧ ∀ (key : K) (b : B),
        Keyed.lookup eqb key m = Some b ->
        (∃ a : A, Keyed.lookup eqb key out = Some a)
        ∨ ∃ s1 s2 : S, P s1 ∧ R s s1 ∧ g key b s1 = Some (None, s2).
Proof.
  intros Hstep s out s' HP Hfold.
  unfold Keyed.filter_map_fold, Keyed.filter_map_fold_on in Hfold.
  set (f :=
    fun (key : K) (s : S) =>
      match Keyed.lookup eqb key m with
      | Some b => g key b s
      | None => Some (None, s)
      end) in Hfold.
  assert (Hstep' :
    ∀ (key : K) (s : S) (a : option A) (s' : S),
      Keyed.among eqb key (Keyed.domain eqb m) = true ->
      P s -> f key s = Some (a, s') -> P s' ∧ R s s').
  { intros key s0 a s0' _ HP0 Hf. unfold f in Hf.
    destruct (Keyed.lookup eqb key m) as [b | ] eqn:Hb.
    - exact (Hstep key b s0 a s0' Hb HP0 Hf).
    - injection Hf as <- <-. split; [ exact HP0 | apply R_refl ]. }
  destruct
    (tabulate_fold_opt_spec f (Keyed.domain eqb m)
       (keyed_domain_distinct K eqb Heqb B m) Hstep' s out s' HP Hfold)
    as (HP' & HR & Hin & Hfound & Hall).
  split; [ exact HP' | ]. split; [ exact HR | ]. split; [ exact Hin | ].
  split.
  - intros key a Ha.
    destruct (Hfound key a Ha) as (_ & s1 & s2 & HP1 & HR1 & Hf & HR2).
    unfold f in Hf. destruct (Keyed.lookup eqb key m) as [b | ];
      [ | discriminate ].
    exists b, s1, s2. repeat split; assumption.
  - intros key b Hb.
    assert (Hamong : Keyed.among eqb key (Keyed.domain eqb m) = true).
    { rewrite (keyed_among_domain K eqb Heqb). unfold Keyed.is_bound.
      rewrite Hb. reflexivity. }
    destruct (Hall key Hamong) as [Ha | (s1 & s2 & HP1 & HR1 & Hf)];
      [ left; exact Ha | right ].
    unfold f in Hf. rewrite Hb in Hf. exists s1, s2. repeat split; assumption.
Qed.

End Fold.

(* ======================================================================== *)
(* The components of products, index by index                               *)
(* ======================================================================== *)

Lemma gather_components_fold {A : Type} (g : positive)
    (rest : list (Index.t * finite_map A)) :
  ∀ acc : finite_map (list (Index.t * A)),
    lookup g
      (List.fold_left
         (fun components_by_index indexed_product2 =>
            merge
              (fun vals_opt val2_opt =>
                 match vals_opt, val2_opt with
                 | None, _ | _, None => None
                 | Some vals, Some val2 =>
                   Some ((fst indexed_product2, val2) :: vals)
                 end)
              components_by_index (snd indexed_product2))
         rest acc)
    = match
        lookup g acc,
        indexed_map_opt (fun _ product => lookup g product) rest
      with
      | Some l, Some r => Some (rev r ++ l)
      | _, _ => None
      end.
Proof.
  induction rest as [ | [id2 p2] rest IH]; intros acc.
  - simpl. destruct (lookup g acc); reflexivity.
  - cbn [List.fold_left indexed_map_opt fst snd].
    rewrite IH, lookup_merge by reflexivity.
    destruct (lookup g acc) as [l | ], (lookup g p2) as [v | ],
      (indexed_map_opt (fun _ product => lookup g product) rest) as [r | ];
      try reflexivity.
    simpl. rewrite <- app_assoc. reflexivity.
Qed.

(* What the gathering binds an index to: the components of the
   products for it, the last product's first, if every product has
   one. *)
Lemma gather_components_spec {A : Type}
    (products : list (Index.t * finite_map A)) (g : positive)
    (components : list (Index.t * A)) :
  lookup g (gather_components products) = Some components ->
  map fst components = rev (map fst products)
  ∧ (∀ (index : Index.t) (product : finite_map A),
       In (index, product) products ->
       ∃ a : A, lookup g product = Some a ∧ In (index, a) components)
  ∧ ∀ (index : Index.t) (a : A),
      In (index, a) components ->
      ∃ product : finite_map A,
        lookup g product = Some a ∧ In (index, product) products.
Proof.
  destruct products as [ | [id1 p1] rest]; [ discriminate | ].
  unfold gather_components.
  rewrite gather_components_fold, lookup_map_values.
  intros Hcomponents.
  assert (Hall :
    ∃ l : list (Index.t * A),
      indexed_map_opt (fun _ product => lookup g product) ((id1, p1) :: rest)
      = Some l
      ∧ components = rev l).
  { cbn [indexed_map_opt].
    destruct (lookup g p1) as [v | ]; [ | discriminate ].
    destruct (indexed_map_opt (fun _ product => lookup g product) rest)
      as [r | ]; [ | discriminate ].
    injection Hcomponents as <-. exists ((id1, v) :: r).
    split; reflexivity. }
  destruct Hall as (l & Hl & ->).
  destruct (indexed_map_opt_spec _ _ l Hl) as (Hfst & Hto & Hfrom).
  split; [ rewrite map_rev, Hfst; reflexivity | ].
  split.
  - intros index product Hin.
    destruct (Hto index product Hin) as (a & Ha & Hin').
    exists a. split; [ exact Ha | apply in_rev in Hin'; exact Hin' ].
  - intros index a Hin. apply in_rev in Hin.
    destruct (Hfrom index a Hin) as (product & Ha & Hin').
    exists product. split; assumption.
Qed.

(* ======================================================================== *)
(* The cases of row-like structures, key by key                             *)
(* ======================================================================== *)

Section Gather_known.

Context {K value other : Type} (eqb : K -> K -> bool)
  (Heqb : decides_equality eqb) (of_other : other -> value).

Local Abbreviation input :=
  (Index.t * (Keyed.finite_map K value * Or_bottom.t other))%type.

(* What a row-like structure gives for a key: its case for the key, or
   failing one its case for the other keys. *)
Definition contribution (key : K) (i : input) : list (Index.t * value) :=
  match Keyed.lookup eqb key (fst (snd i)) with
  | Some v => [(fst i, v)]
  | None =>
    match snd (snd i) with
    | Or_bottom.Ok o => [(fst i, of_other o)]
    | Or_bottom.Bottom => []
    end
  end.

Definition other_of (i : input) : list (Index.t * value) :=
  match snd (snd i) with
  | Or_bottom.Ok o => [(fst i, of_other o)]
  | Or_bottom.Bottom => []
  end.

Definition bound_in_some (key : K) (inputs : list input) : bool :=
  existsb (fun i => Keyed.is_bound eqb key (fst (snd i))) inputs.

Lemma contributions_unbound (key : K) (inputs : list input) :
  bound_in_some key inputs = false ->
  flat_map (contribution key) inputs = flat_map other_of inputs.
Proof.
  induction inputs as [ | i inputs IH]; simpl; [ reflexivity | ].
  intros Hbound. apply orb_false_iff in Hbound as [Hi Hrest].
  rewrite (IH Hrest). f_equal. unfold contribution, Keyed.is_bound in *.
  destruct (Keyed.lookup eqb key (fst (snd i))); [ discriminate | reflexivity ].
Qed.

Lemma add_other_eq (acc : list (Index.t * value)) (i : input) :
  add_other of_other acc (fst i) (snd (snd i)) = rev (other_of i) ++ acc.
Proof.
  unfold add_other, other_of. destruct (snd (snd i)); reflexivity.
Qed.

Local Abbreviation gather_step :=
  (fun (acc :
          Keyed.finite_map K (list (Index.t * value))
          * list (Index.t * value)) (i : input) =>
     let id2 := fst i in
     let map2 := fst (snd i) in
     let other2 := snd (snd i) in
     (Keyed.merge eqb
        (fun vals_opt val2_opt =>
           match vals_opt, val2_opt with
           | None, None => None
           | None, Some val2 => Some ((id2, val2) :: snd acc)
           | Some vals, None => Some (add_other of_other vals id2 other2)
           | Some vals, Some val2 => Some ((id2, val2) :: vals)
           end)
        (fst acc) map2,
      add_other of_other (snd acc) id2 other2)).

Lemma gather_known_fold (rest : list input) :
  ∀ (done : list input)
    (acc :
       Keyed.finite_map K (list (Index.t * value)) * list (Index.t * value)),
    (∀ key : K,
       Keyed.lookup eqb key (fst acc)
       = if bound_in_some key done
         then Some (rev (flat_map (contribution key) done))
         else None) ->
    snd acc = rev (flat_map other_of done) ->
    ∀ key : K,
      Keyed.lookup eqb key (fst (List.fold_left gather_step rest acc))
      = if bound_in_some key (done ++ rest)
        then Some (rev (flat_map (contribution key) (done ++ rest)))
        else None.
Proof.
  induction rest as [ | i rest IH]; intros done acc Hmaps Hothers key.
  - rewrite app_nil_r. apply Hmaps.
  - cbn [List.fold_left].
    replace (done ++ i :: rest) with ((done ++ [i]) ++ rest)
      by (rewrite <- app_assoc; reflexivity).
    apply IH.
    + intros key'. cbn [fst snd].
      rewrite (keyed_lookup_merge K eqb Heqb) by reflexivity.
      rewrite Hmaps. unfold bound_in_some.
      rewrite existsb_app, flat_map_app, rev_app_distr. cbn [existsb flat_map].
      rewrite orb_false_r, app_nil_r.
      fold (bound_in_some key' done).
      destruct (bound_in_some key' done) eqn:Hdone.
      * cbn [orb]. unfold contribution.
        destruct (Keyed.lookup eqb key' (fst (snd i))) as [val2 | ];
          [ reflexivity | ].
        rewrite add_other_eq. reflexivity.
      * cbn [orb]. rewrite (contributions_unbound key' done Hdone), <- Hothers.
        unfold contribution, Keyed.is_bound.
        destruct (Keyed.lookup eqb key' (fst (snd i))) as [val2 | ];
          reflexivity.
    + cbn [fst snd]. rewrite add_other_eq, Hothers, flat_map_app, rev_app_distr.
      cbn [flat_map]. rewrite app_nil_r. reflexivity.
Qed.

(* What the gathering binds a key to: what each row-like structure
   gives for it, the last one's first, if one of them has a case for
   it. *)
Lemma gather_known_lookup (inputs : list input) (key : K) :
  Keyed.lookup eqb key (gather_known eqb of_other inputs)
  = if bound_in_some key inputs
    then Some (rev (flat_map (contribution key) inputs))
    else None.
Proof.
  destruct inputs as [ | i1 rest]; [ reflexivity | ].
  destruct i1 as [id1 [map1 other1]]. unfold gather_known.
  apply (gather_known_fold rest [(id1, (map1, other1))]).
  - intros key'. cbn [fst snd].
    rewrite keyed_lookup_map_values. unfold bound_in_some.
    cbn [existsb flat_map fst snd]. rewrite orb_false_r, app_nil_r.
    unfold contribution, Keyed.is_bound. cbn [fst snd].
    destruct (Keyed.lookup eqb key' map1); reflexivity.
  - cbn [fst snd flat_map]. rewrite app_nil_r.
    change (add_other of_other [] id1 other1)
      with (add_other of_other [] (fst (id1, (map1, other1)))
              (snd (snd (id1, (map1, other1))))).
    rewrite add_other_eq, app_nil_r. reflexivity.
Qed.

Lemma contribution_fst (key : K) (i : input) (index : Index.t) (v : value) :
  In (index, v) (contribution key i) -> index = fst i.
Proof.
  unfold contribution.
  destruct (Keyed.lookup eqb key (fst (snd i)));
    [ intros [[= <- _] | []]; reflexivity | ].
  destruct (snd (snd i)); [ | intros [] ].
  intros [[= <- _] | []]. reflexivity.
Qed.

Lemma contributions_distinct (key : K) (inputs : list input) :
  NoDup (map fst inputs) ->
  NoDup (map fst (flat_map (contribution key) inputs)).
Proof.
  induction inputs as [ | i inputs IH]; simpl; intros Hdistinct;
    [ apply NoDup_nil | ].
  inversion Hdistinct as [ | ? ? Hi Hrest]. subst.
  rewrite map_app.
  assert (Hfresh : ¬ In (fst i) (map fst (flat_map (contribution key) inputs))).
  { intros Hin. apply Hi. apply in_map_iff in Hin as ([index v] & Heq & Hin).
    simpl in Heq. subst index.
    apply in_flat_map in Hin as (i' & Hin' & Hv).
    apply contribution_fst in Hv. rewrite Hv. apply in_map. exact Hin'. }
  unfold contribution at 1.
  destruct (Keyed.lookup eqb key (fst (snd i))).
  - simpl. apply NoDup_cons; [ exact Hfresh | exact (IH Hrest) ].
  - destruct (snd (snd i)); simpl;
      [ apply NoDup_cons; [ exact Hfresh | exact (IH Hrest) ]
      | exact (IH Hrest) ].
Qed.

(* What the cases gathered for a key are, by membership. *)
Lemma gather_known_spec (inputs : list input) (key : K)
    (vals : list (Index.t * value)) :
  NoDup (map fst inputs) ->
  Keyed.lookup eqb key (gather_known eqb of_other inputs) = Some vals ->
  NoDup (map fst vals)
  ∧ (∀ (index : Index.t) (v : value),
       In (index, v) vals ->
       ∃ i : input,
         In i inputs ∧ fst i = index
         ∧ (Keyed.lookup eqb key (fst (snd i)) = Some v
            ∨ (Keyed.lookup eqb key (fst (snd i)) = None
               ∧ ∃ o : other,
                   snd (snd i) = Or_bottom.Ok o ∧ v = of_other o)))
  ∧ (∀ i : input,
       In i inputs ->
       (∀ v : value,
          Keyed.lookup eqb key (fst (snd i)) = Some v -> In (fst i, v) vals)
       ∧ (Keyed.lookup eqb key (fst (snd i)) = None ->
          ∀ o : other,
            snd (snd i) = Or_bottom.Ok o -> In (fst i, of_other o) vals))
  ∧ vals <> [].
Proof.
  intros Hdistinct. rewrite gather_known_lookup.
  destruct (bound_in_some key inputs) eqn:Hbound; [ | discriminate ].
  intros [= <-].
  split; [ rewrite map_rev; apply NoDup_rev, contributions_distinct;
           exact Hdistinct | ].
  assert (Hto :
    ∀ i : input,
      In i inputs ->
      (∀ v : value,
         Keyed.lookup eqb key (fst (snd i)) = Some v ->
         In (fst i, v) (rev (flat_map (contribution key) inputs)))
      ∧ (Keyed.lookup eqb key (fst (snd i)) = None ->
         ∀ o : other,
           snd (snd i) = Or_bottom.Ok o ->
           In (fst i, of_other o) (rev (flat_map (contribution key) inputs)))).
  { intros i Hin. split.
    - intros v Hv. rewrite <- in_rev. apply in_flat_map. exists i.
      split; [ exact Hin | ]. unfold contribution. rewrite Hv. left.
      reflexivity.
    - intros Hnone o Ho. rewrite <- in_rev. apply in_flat_map. exists i.
      split; [ exact Hin | ]. unfold contribution. rewrite Hnone, Ho. left.
      reflexivity. }
  split; [ | split; [ exact Hto | ] ].
  - intros index v Hin. apply in_rev in Hin.
    apply in_flat_map in Hin as (i & Hin & Hv). exists i.
    split; [ exact Hin | ].
    split; [ symmetry; exact (contribution_fst key i index v Hv) | ].
    unfold contribution in Hv.
    destruct (Keyed.lookup eqb key (fst (snd i))) as [v' | ].
    + destruct Hv as [[= _ <-] | []]. left. reflexivity.
    + right. split; [ reflexivity | ].
      destruct (snd (snd i)) as [o | ]; [ | destruct Hv ].
      destruct Hv as [[= _ <-] | []]. exists o. split; reflexivity.
  - unfold bound_in_some in Hbound.
    apply existsb_exists in Hbound as (i & Hin & Hi).
    apply (keyed_is_bound_lookup eqb key) in Hi as [v Hv].
    intros Hnil. pose proof (proj1 (Hto i Hin) v Hv) as Hin'.
    rewrite Hnil in Hin'. destruct Hin'.
Qed.

(* A key that the gathering does not bind has no case in any row-like
   structure. *)
Lemma gather_known_none (inputs : list input) (key : K) :
  Keyed.lookup eqb key (gather_known eqb of_other inputs) = None ->
  ∀ i : input, In i inputs -> Keyed.lookup eqb key (fst (snd i)) = None.
Proof.
  rewrite gather_known_lookup.
  destruct (bound_in_some key inputs) eqn:Hbound; [ discriminate | ].
  intros _ i Hin. unfold bound_in_some in Hbound.
  destruct (Keyed.lookup eqb key (fst (snd i))) as [v | ] eqn:Hv;
    [ | reflexivity ].
  assert (Htrue :
    existsb (fun i => Keyed.is_bound eqb key (fst (snd i))) inputs = true).
  { apply existsb_exists. exists i. split; [ exact Hin | ].
    unfold Keyed.is_bound. rewrite Hv. reflexivity. }
  congruence.
Qed.

End Gather_known.

(* ======================================================================== *)
(* Lists made of lists with indices                                         *)
(* ======================================================================== *)

(* A list made of at most one thing for each element of a list whose
   indices are distinct has distinct indices. *)
Lemma flat_map_indexed_distinct {A B : Type}
    (f : Index.t * A -> list (Index.t * B)) (l : list (Index.t * A)) :
  (∀ (i : Index.t * A) (index : Index.t) (b : B),
     In (index, b) (f i) -> index = fst i) ->
  (∀ i : Index.t * A, NoDup (map fst (f i))) ->
  NoDup (map fst l) -> NoDup (map fst (flat_map f l)).
Proof.
  intros Hfst Hone. induction l as [ | i l IH]; simpl; intros Hdistinct;
    [ apply NoDup_nil | ].
  inversion Hdistinct as [ | ? ? Hi Hrest]. subst.
  rewrite map_app. apply NoDup_app; [ apply Hone | exact (IH Hrest) | ].
  intros index Hin Hin'. apply Hi.
  apply in_map_iff in Hin as ([index1 b1] & Heq1 & Hin1). simpl in Heq1.
  subst index1. apply Hfst in Hin1.
  apply in_map_iff in Hin' as ([index2 b2] & Heq2 & Hin2). simpl in Heq2.
  subst index2. apply in_flat_map in Hin2 as (i' & Hi' & Hin2).
  apply Hfst in Hin2. rewrite <- Hin1, Hin2. apply in_map. exact Hi'.
Qed.

Lemma filter_indexed_distinct {A : Type} (keep : Index.t * A -> bool)
    (l : list (Index.t * A)) :
  NoDup (map fst l) -> NoDup (map fst (filter keep l)).
Proof.
  induction l as [ | i l IH]; simpl; intros Hdistinct; [ apply NoDup_nil | ].
  inversion Hdistinct as [ | ? ? Hi Hrest]. subst.
  destruct (keep i); [ | exact (IH Hrest) ].
  simpl. apply NoDup_cons; [ | exact (IH Hrest) ].
  intros Hin. apply Hi. apply in_map_iff in Hin as (i' & Heq & Hin).
  apply filter_In in Hin as [Hin _]. rewrite <- Heq. apply in_map. exact Hin.
Qed.

(* The things of a list that are not bottom. *)
Definition ok_of {A : Type} (x : Index.t * Or_bottom.t A)
    : list (Index.t * A) :=
  match snd x with
  | Or_bottom.Ok a => [(fst x, a)]
  | Or_bottom.Bottom => []
  end.

Lemma non_bottom_eq {A : Type} (l : list (Index.t * Or_bottom.t A)) :
  ∀ acc : list (Index.t * A), non_bottom acc l = rev (flat_map ok_of l) ++ acc.
Proof.
  induction l as [ | [id [a | ]] l IH]; intros acc; simpl.
  - reflexivity.
  - rewrite IH. unfold ok_of at 2. simpl. rewrite <- app_assoc. reflexivity.
  - rewrite IH. reflexivity.
Qed.

Lemma non_bottom_spec {A : Type} (l : list (Index.t * Or_bottom.t A)) :
  NoDup (map fst l) ->
  NoDup (map fst (non_bottom [] l))
  ∧ ∀ (index : Index.t) (a : A),
      In (index, a) (non_bottom [] l) <-> In (index, Or_bottom.Ok a) l.
Proof.
  intros Hdistinct. rewrite non_bottom_eq, app_nil_r.
  split.
  - rewrite map_rev. apply NoDup_rev.
    apply flat_map_indexed_distinct; [ | | exact Hdistinct ].
    + intros [id [a | ]] index b; unfold ok_of; simpl;
        [ intros [[= <- _] | []]; reflexivity | intros [] ].
    + intros [id [a | ]]; unfold ok_of; simpl;
        [ apply NoDup_cons; [ intros [] | apply NoDup_nil ] | apply NoDup_nil ].
  - intros index a. rewrite <- in_rev, in_flat_map. split.
    + intros ([id [a' | ]] & Hin & Hok); unfold ok_of in Hok; simpl in Hok;
        [ | destruct Hok ].
      destruct Hok as [[= <- <-] | []]. exact Hin.
    + intros Hin. exists (index, Or_bottom.Ok a).
      split; [ exact Hin | left; reflexivity ].
Qed.

(* ======================================================================== *)
(* The index of the join of cases                                           *)
(* ======================================================================== *)

Section Index.

(* What the index of a case admits is [W]: its domain admits it exactly
   or from below, and its shape admits it. *)
Context {lattice shape W : Type} (equal_index : lattice -> lattice -> bool)
  (inter_index : lattice -> lattice -> lattice)
  (n_way_join_shape : list shape -> or_unknown_or_bottom shape)
  (exact at_least : W -> lattice -> Prop) (shape_holds : W -> shape -> Prop).

Definition domain_holds (w : W) (domain : Row_like_index_domain.t lattice)
    : Prop :=
  match domain with
  | Row_like_index_domain.Known l => exact w l
  | Row_like_index_domain.At_least l => at_least w l
  end.

Hypothesis Hexact : ∀ (w : W) (l : lattice), exact w l -> at_least w l.
Hypothesis Hequal :
  ∀ (w : W) (l l' : lattice),
    equal_index l l' = true -> exact w l' -> exact w l.
Hypothesis Hinter_left :
  ∀ (w : W) (l l' : lattice), at_least w l -> at_least w (inter_index l l').
Hypothesis Hinter_right :
  ∀ (w : W) (l l' : lattice), at_least w l' -> at_least w (inter_index l l').
Hypothesis Hshapes :
  ∀ (shapes : list shape) (joined : shape),
    n_way_join_shape shapes = Ok joined ->
    ∀ shape' : shape,
      In shape' shapes ->
      ∀ w : W, shape_holds w shape' -> shape_holds w joined.
Hypothesis Hshapes_bottom :
  ∀ shapes : list shape, n_way_join_shape shapes = Bottom -> shapes = [].

Local Abbreviation inter_step :=
  (fun (i1' : lattice) (i2 : row_like_index lattice shape) =>
     match index_domain i2 with
     | Row_like_index_domain.At_least i2'
     | Row_like_index_domain.Known i2' => inter_index i1' i2'
     end).

Lemma fold_inter_holds (indexes : list (row_like_index lattice shape)) :
  ∀ (l : lattice) (w : W),
    at_least w l -> at_least w (List.fold_left inter_step indexes l).
Proof.
  induction indexes as [ | i indexes IH]; intros l w Hl; simpl;
    [ exact Hl | ].
  apply IH. destruct (index_domain i); apply Hinter_left; exact Hl.
Qed.

Lemma fold_inter_in (indexes : list (row_like_index lattice shape)) :
  ∀ (l : lattice) (w : W) (i : row_like_index lattice shape),
    In i indexes ->
    domain_holds w (index_domain i) ->
    at_least w (List.fold_left inter_step indexes l).
Proof.
  induction indexes as [ | i' indexes IH]; intros l w i Hin Hi; simpl;
    [ destruct Hin | ].
  destruct Hin as [-> | Hin]; [ | exact (IH _ w i Hin Hi) ].
  apply fold_inter_holds. unfold domain_holds in Hi.
  destruct (index_domain i);
    [ apply Hinter_right, Hexact; exact Hi | apply Hinter_right; exact Hi ].
Qed.

(* What the index of each case admits, the index of the join admits. *)
Lemma n_way_join_index_sound (indexes : list (row_like_index lattice shape))
    (index : row_like_index lattice shape) :
  n_way_join_index equal_index inter_index n_way_join_shape indexes
  = Ok index ->
  ∀ i : row_like_index lattice shape,
    In i indexes ->
    ∀ w : W,
      domain_holds w (index_domain i) ->
      shape_holds w (index_shape i) ->
      domain_holds w (index_domain index) ∧ shape_holds w (index_shape index).
Proof.
  unfold n_way_join_index. intros Hjoined i Hin w Hdomain Hshape.
  destruct (n_way_join_shape (map index_shape indexes)) as [ | joined | ]
    eqn:Hjoined_shape; try discriminate.
  assert (Hshape' : shape_holds w joined).
  { apply (Hshapes _ joined Hjoined_shape (index_shape i)); [ | exact Hshape ].
    apply in_map. exact Hin. }
  destruct indexes as [ | i1 rest]; [ discriminate | ].
  assert (Hat_least :
    ∀ i1' : lattice,
      (i = i1 -> at_least w i1') ->
      at_least w (List.fold_left inter_step rest i1')).
  { intros i1' Hfirst. destruct Hin as [<- | Hin].
    - apply fold_inter_holds, Hfirst. reflexivity.
    - exact (fold_inter_in rest i1' w i Hin Hdomain). }
  destruct (index_domain i1) as [i1' | i1'] eqn:Hdomain1.
  - destruct
      (forallb
         (fun i2 =>
            match index_domain i2 with
            | Row_like_index_domain.Known i2' => equal_index i1' i2'
            | Row_like_index_domain.At_least _ => false
            end)
         rest) eqn:Hall.
    + injection Hjoined as <-. simpl. split; [ | exact Hshape' ].
      destruct Hin as [<- | Hin].
      * rewrite Hdomain1 in Hdomain. exact Hdomain.
      * rewrite forallb_forall in Hall. specialize (Hall i Hin).
        unfold domain_holds in Hdomain.
        destruct (index_domain i) as [i2' | i2']; [ | discriminate ].
        exact (Hequal w i1' i2' Hall Hdomain).
    + injection Hjoined as <-. simpl. split; [ | exact Hshape' ].
      apply Hat_least. intros ->. rewrite Hdomain1 in Hdomain.
      apply Hexact. exact Hdomain.
  - injection Hjoined as <-. simpl. split; [ | exact Hshape' ].
    apply Hat_least. intros ->. rewrite Hdomain1 in Hdomain. exact Hdomain.
Qed.

(* The join of the indices of some cases is not bottom. *)
Lemma n_way_join_index_bottom (indexes : list (row_like_index lattice shape)) :
  n_way_join_index equal_index inter_index n_way_join_shape indexes = Bottom ->
  indexes = [].
Proof.
  unfold n_way_join_index. intros Hjoined.
  destruct (n_way_join_shape (map index_shape indexes)) as [ | joined | ]
    eqn:Hjoined_shape; try discriminate.
  - destruct indexes as [ | i1 rest]; [ reflexivity | ].
    destruct (index_domain i1); [ | discriminate ].
    destruct (forallb _ rest); discriminate.
  - apply Hshapes_bottom in Hjoined_shape.
    exact (map_eq_nil _ _ Hjoined_shape).
Qed.

End Index.

(* ======================================================================== *)
(* The parts of heads, gathered                                             *)
(* ======================================================================== *)

Lemma kind_eqb_true (k k' : flambda_kind) : kind_eqb k k' = true -> k = k'.
Proof.
  unfold kind_eqb. destruct (kind_eq_dec k k'); [ auto | discriminate ].
Qed.

Lemma kind_eqb_refl (k : flambda_kind) : kind_eqb k k = true.
Proof.
  unfold kind_eqb. destruct (kind_eq_dec k k); [ reflexivity | contradiction ].
Qed.

Lemma snoc_distinct {A : Type} (a : A) (l : list A) :
  NoDup (a :: l) -> NoDup (l ++ [a]).
Proof.
  intros Hdistinct. inversion Hdistinct as [ | ? ? Ha Hl]. subst.
  apply NoDup_app;
    [ exact Hl | apply NoDup_cons; [ intros [] | apply NoDup_nil ] | ].
  intros x Hx [<- | []]. exact (Ha Hx).
Qed.

(* The fold over the heads that follow the first is a fold over what
   is read of them, if each can be read. *)
Lemma gather_heads_eq {A B : Type}
    (read : head_of_kind_value_non_null -> option A)
    (combine : Index.t -> A -> B -> B)
    (other_heads : list (Index.t * head_of_kind_value_non_null)) (init : B) :
  gather_heads read combine other_heads init
  = option_map
      (fun l =>
         List.fold_right (fun x acc => combine (fst x) (snd x) acc) init l)
      (indexed_map_opt (fun _ head => read head) other_heads).
Proof.
  induction other_heads as [ | [id head] other_heads IH]; simpl;
    [ reflexivity | ].
  rewrite IH.
  destruct (indexed_map_opt (fun _ head => read head) other_heads) as [l | ];
    destruct (read head) as [a | ]; reflexivity.
Qed.

(* The parts gathered of heads that have a part to join and an
   allocation mode. *)
Lemma gathered_pairs_fst {A M : Type} (join : M -> M -> M)
    (l : list (Index.t * (A * M))) (init : list (Index.t * A)) (m : M) :
  fst
    (List.fold_right
       (fun x acc =>
          ((fst x, fst (snd x)) :: fst acc, join (snd acc) (snd (snd x))))
       (init, m) l)
  = map (fun x => (fst x, fst (snd x))) l ++ init.
Proof.
  induction l as [ | x l IH]; simpl; [ reflexivity | rewrite IH; reflexivity ].
Qed.

Lemma strings_union_in (bytes : list Z) (strs strs' : list (list Z)) :
  bytes ∈ strings_union strs strs' <-> bytes ∈ strs ∨ bytes ∈ strs'.
Proof.
  unfold strings_union.
  rewrite <-
    (keyed_among_in (list Z) (lists_eqb Z.eqb) bytes_eqb_eq bytes
       (Keyed.dedup (lists_eqb Z.eqb) _)).
  rewrite (keyed_among_dedup (list Z) (lists_eqb Z.eqb) bytes_eqb_eq).
  rewrite (keyed_among_in (list Z) (lists_eqb Z.eqb) bytes_eqb_eq).
  apply in_app_iff.
Qed.

Lemma fold_strings_keeps (l : list (Index.t * list (list Z)))
    (init : list (list Z)) (bytes : list Z) :
  bytes ∈ init ->
  bytes
  ∈ List.fold_right (fun x acc => strings_union (snd x) acc) init l.
Proof.
  intros Hin. induction l as [ | x l IH]; simpl; [ exact Hin | ].
  apply strings_union_in. right. exact IH.
Qed.

Lemma fold_strings_in (l : list (Index.t * list (list Z)))
    (init : list (list Z)) (index : Index.t) (strs : list (list Z))
    (bytes : list Z) :
  In (index, strs) l ->
  bytes ∈ strs ->
  bytes
  ∈ List.fold_right (fun x acc => strings_union (snd x) acc) init l.
Proof.
  intros Hin Hbytes. induction l as [ | x l IH]; simpl; [ destruct Hin | ].
  apply strings_union_in. destruct Hin as [-> | Hin];
    [ left; exact Hbytes | right; exact (IH Hin) ].
Qed.

(* The extensions of variants, if each has them. *)
Lemma gather_extensions_spec
    (extensions : list (Index.t * variant_extensions)) :
  ∀ (acc_i acc_b when_immediates when_blocks : list (Index.t * env_extension)),
    gather_extensions acc_i acc_b extensions
    = Some (when_immediates, when_blocks) ->
    map fst when_immediates = rev (map fst extensions) ++ map fst acc_i
    ∧ map fst when_blocks = rev (map fst extensions) ++ map fst acc_b
    ∧ (∀ (index : Index.t) (e : variant_extensions),
         In (index, e) extensions ->
         ∃ wi wb : env_extension,
           e = Ext wi wb ∧ In (index, wi) when_immediates
           ∧ In (index, wb) when_blocks)
    ∧ (∀ (index : Index.t) (wi : env_extension),
         In (index, wi) when_immediates ->
         In (index, wi) acc_i
         ∨ ∃ wb : env_extension, In (index, Ext wi wb) extensions)
    ∧ (∀ (index : Index.t) (wb : env_extension),
         In (index, wb) when_blocks ->
         In (index, wb) acc_b
         ∨ ∃ wi : env_extension, In (index, Ext wi wb) extensions)
    ∧ (∀ (index : Index.t) (wi : env_extension),
         In (index, wi) acc_i -> In (index, wi) when_immediates)
    ∧ ∀ (index : Index.t) (wb : env_extension),
        In (index, wb) acc_b -> In (index, wb) when_blocks.
Proof.
  induction extensions as [ | [id [ | wi0 wb0]] extensions IH];
    intros acc_i acc_b when_immediates when_blocks Hgather;
    simpl in Hgather.
  - injection Hgather as <- <-. simpl.
    split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ intros ? ? [] | ]. split; [ auto | ]. split; [ auto | ].
    split; auto.
  - discriminate.
  - destruct (IH _ _ _ _ Hgather)
      as (Hfst_i & Hfst_b & Hto & Hfrom_i & Hfrom_b & Hacc_i & Hacc_b).
    split; [ rewrite Hfst_i; simpl; rewrite <- app_assoc; reflexivity | ].
    split; [ rewrite Hfst_b; simpl; rewrite <- app_assoc; reflexivity | ].
    split; [ | split; [ | split; [ | split ] ] ].
    + intros index e [[= <- <-] | Hin]; [ | exact (Hto index e Hin) ].
      exists wi0, wb0. split; [ reflexivity | ].
      split; [ apply Hacc_i | apply Hacc_b ]; left; reflexivity.
    + intros index wi Hin.
      destruct (Hfrom_i index wi Hin) as [[[= <- <-] | Hin'] | [wb Hin']].
      * right. exists wb0. left. reflexivity.
      * left. exact Hin'.
      * right. exists wb. right. exact Hin'.
    + intros index wb Hin.
      destruct (Hfrom_b index wb Hin) as [[[= <- <-] | Hin'] | [wi Hin']].
      * right. exists wi0. left. reflexivity.
      * left. exact Hin'.
      * right. exists wi. right. exact Hin'.
    + intros index wi Hin. apply Hacc_i. right. exact Hin.
    + intros index wb Hin. apply Hacc_b. right. exact Hin.
Qed.

(* The heads that are not null, if none is unknown. *)
Definition ok_head (x : Index.t * head_of_kind_value)
    : list (Index.t * head_of_kind_value_non_null) :=
  match snd x with
  | Head_of_kind_value (Ok head) _ => [(fst x, head)]
  | Head_of_kind_value _ _ => []
  end.

Lemma non_null_heads_spec (heads : list (Index.t * head_of_kind_value)) :
  ∀ acc non_nulls : list (Index.t * head_of_kind_value_non_null),
    non_null_heads acc heads = Some non_nulls ->
    non_nulls = rev (flat_map ok_head heads) ++ acc
    ∧ ∀ (index : Index.t)
        (non_null : or_unknown_or_bottom head_of_kind_value_non_null)
        (is_null : is_null),
        In (index, Head_of_kind_value non_null is_null) heads ->
        non_null <> Unknown.
Proof.
  induction heads as [ | [id [non_null is_null]] heads IH];
    intros acc non_nulls Hheads; simpl in Hheads.
  - injection Hheads as <-. split; [ reflexivity | intros ? ? ? [] ].
  - destruct non_null as [ | head | ]; [ discriminate | | ].
    + destruct (IH _ _ Hheads) as (-> & Hknown). split.
      * simpl. unfold ok_head at 2. simpl. rewrite <- app_assoc. reflexivity.
      * intros index non_null' is_null' [[= <- <- <-] | Hin];
          [ discriminate | exact (Hknown _ _ _ Hin) ].
    + destruct (IH _ _ Hheads) as (-> & Hknown).
      split; [ reflexivity | ].
      intros index non_null' is_null' [[= <- <- <-] | Hin];
        [ discriminate | exact (Hknown _ _ _ Hin) ].
Qed.

(* The heads of expanded types, those that are bottom left out, if
   none is unknown and each is of the sort asked. *)
Definition extracted {A : Type} (extract : Expanded_type.head -> option A)
    (x : Index.t * Expanded_type.t) : list (Index.t * A) :=
  match Expanded_type.descr (snd x) with
  | Ok head =>
    match extract head with
    | Some a => [(fst x, a)]
    | None => []
    end
  | Unknown | Bottom => []
  end.

Lemma extract_heads_spec {A : Type} (extract : Expanded_type.head -> option A)
    (expandeds : list (Index.t * Expanded_type.t)) :
  ∀ heads : list (Index.t * A),
    extract_heads extract expandeds = Some (Some heads) ->
    heads = flat_map (extracted extract) expandeds
    ∧ ∀ (index : Index.t) (e : Expanded_type.t),
        In (index, e) expandeds ->
        Expanded_type.descr e = Bottom
        ∨ ∃ (head : Expanded_type.head) (a : A),
            Expanded_type.descr e = Ok head ∧ extract head = Some a.
Proof.
  induction expandeds as [ | [id e] expandeds IH]; intros heads Hheads;
    simpl in Hheads.
  - injection Hheads as <-. split; [ reflexivity | intros ? ? [] ].
  - destruct (Expanded_type.descr e) as [ | head | ] eqn:Hdescr;
      [ discriminate | | ].
    + destruct (extract head) as [a | ] eqn:Ha; [ | discriminate ].
      destruct (extract_heads extract expandeds) as [[rest | ] | ];
        try discriminate.
      simpl in Hheads. injection Hheads as <-.
      destruct (IH rest eq_refl) as (-> & Hall). split.
      * cbn [flat_map].
        replace (extracted extract (id, e)) with [(id, a)];
          [ reflexivity | ].
        unfold extracted. cbn [fst snd]. rewrite Hdescr, Ha. reflexivity.
      * intros index e' [[= <- <-] | Hin];
          [ right; exists head, a; split; assumption | exact (Hall _ _ Hin) ].
    + destruct (IH heads Hheads) as (-> & Hall). split.
      * cbn [flat_map].
        replace (extracted extract (id, e)) with (@nil (Index.t * A));
          [ reflexivity | ].
        unfold extracted. cbn [fst snd]. rewrite Hdescr. reflexivity.
      * intros index e' [[= <- <-] | Hin];
          [ left; exact Hdescr | exact (Hall _ _ Hin) ].
Qed.

Lemma extracted_in {A : Type} (extract : Expanded_type.head -> option A)
    (expandeds : list (Index.t * Expanded_type.t)) (index : Index.t) (a : A) :
  In (index, a) (flat_map (extracted extract) expandeds)
  <-> ∃ (e : Expanded_type.t) (head : Expanded_type.head),
        In (index, e) expandeds
        ∧ Expanded_type.descr e = Ok head ∧ extract head = Some a.
Proof.
  rewrite in_flat_map. split.
  - intros ([id e] & Hin & Ha). unfold extracted in Ha. simpl in Ha.
    destruct (Expanded_type.descr e) as [ | head | ] eqn:Hdescr;
      [ destruct Ha | | destruct Ha ].
    destruct (extract head) as [a' | ] eqn:Ha'; [ | destruct Ha ].
    destruct Ha as [[= <- <-] | []]. exists e, head. auto.
  - intros (e & head & Hin & Hdescr & Ha). exists (index, e).
    split; [ exact Hin | ]. unfold extracted. simpl. rewrite Hdescr, Ha.
    left. reflexivity.
Qed.

Lemma extracted_distinct {A : Type} (extract : Expanded_type.head -> option A)
    (expandeds : list (Index.t * Expanded_type.t)) :
  NoDup (map fst expandeds) ->
  NoDup (map fst (flat_map (extracted extract) expandeds)).
Proof.
  apply flat_map_indexed_distinct.
  - intros [id e] index a. unfold extracted. simpl.
    destruct (Expanded_type.descr e) as [ | head | ];
      [ intros [] | | intros [] ].
    destruct (extract head);
      [ intros [[= <- _] | []]; reflexivity | intros [] ].
  - intros [id e]. unfold extracted. simpl.
    destruct (Expanded_type.descr e) as [ | head | ]; try apply NoDup_nil.
    destruct (extract head); [ | apply NoDup_nil ].
    apply NoDup_cons; [ intros [] | apply NoDup_nil ].
Qed.

(* The parts of variants. *)
Lemma read_variant_eq (head : head_of_kind_value_non_null)
    (fields : variant_fields) :
  read_variant head = Some fields ->
  head
  = Variant (variant_is_int fields) (variant_immediates fields)
      (variant_get_tag fields) (variant_blocks fields)
      (variant_extensions_of fields) (variant_is_unique fields).
Proof. destruct head; try discriminate. intros [= <-]. reflexivity. Qed.

Definition is_int_simple_of (fields : variant_fields) : option simple :=
  deduce_is_int_simple (variant_blocks fields) (variant_immediates fields)
    (variant_is_int fields).

Definition get_tag_simple_of (fields : variant_fields)
    : or_unknown_or_bottom simple :=
  deduce_get_tag_simple (variant_blocks fields) (variant_get_tag fields).

Definition get_tag_simples_of (x : Index.t * variant_fields)
    : list (Index.t * simple) :=
  match get_tag_simple_of (snd x) with
  | Ok simple => [(fst x, simple)]
  | Unknown | Bottom => []
  end.

Definition variant_parts_of (first_id : Index.t) (first : variant_fields)
    (l : list (Index.t * variant_fields)) : variant_parts :=
  List.fold_right (fun x acc => add_variant_parts (fst x) (snd x) acc)
    (first_variant_parts first_id first) l.

Lemma variant_parts_spec (first_id : Index.t) (first : variant_fields)
    (l : list (Index.t * variant_fields)) :
  parts_blocks (variant_parts_of first_id first l)
  = map (fun x => (fst x, variant_blocks (snd x))) (l ++ [(first_id, first)])
  ∧ parts_immediates (variant_parts_of first_id first l)
    = map (fun x => (fst x, variant_immediates (snd x)))
        (l ++ [(first_id, first)])
  ∧ parts_extensions (variant_parts_of first_id first l)
    = map (fun x => (fst x, variant_extensions_of (snd x)))
        (l ++ [(first_id, first)])
  ∧ (∀ simples : list (Index.t * simple),
       parts_is_int (variant_parts_of first_id first l) = Some simples ->
       indexed_map_opt (fun _ fields => is_int_simple_of fields)
         (l ++ [(first_id, first)])
       = Some simples)
  ∧ ∀ simples : list (Index.t * simple),
      parts_get_tag (variant_parts_of first_id first l) = Some simples ->
      simples = flat_map get_tag_simples_of (l ++ [(first_id, first)])
      ∧ ∀ x : Index.t * variant_fields,
          In x (l ++ [(first_id, first)]) ->
          get_tag_simple_of (snd x) <> Unknown.
Proof.
  induction l as [ | [id fields] l IH].
  - unfold variant_parts_of. simpl.
    split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ reflexivity | ].
    split.
    + intros simples. unfold is_int_simple_of.
      destruct (deduce_is_int_simple _ _ _);
        [ intros [= <-]; reflexivity | discriminate ].
    + intros simples. unfold get_tag_simples_of, get_tag_simple_of. simpl.
      destruct (deduce_get_tag_simple _ _) eqn:Htag;
        [ discriminate | intros [= <-] | intros [= <-] ];
        (split;
         [ reflexivity
         | intros x [<- | []]; simpl; rewrite Htag; discriminate ]).
  - destruct IH as (Hblocks & Himmediates & Hextensions & His_int & Hget_tag).
    unfold variant_parts_of in *. cbn [List.fold_right app fst snd].
    set (parts :=
      List.fold_right (fun x acc => add_variant_parts (fst x) (snd x) acc)
        (first_variant_parts first_id first) l) in *.
    split; [ simpl; rewrite Hblocks; reflexivity | ].
    split; [ simpl; rewrite Himmediates; reflexivity | ].
    split; [ simpl; rewrite Hextensions; reflexivity | ].
    split.
    + intros simples. cbn [add_variant_parts parts_is_int].
      destruct (parts_is_int parts) as [is_int | ]; [ | discriminate ].
      destruct (deduce_is_int_simple _ _ _) as [other_is_int | ] eqn:Hother;
        [ | discriminate ].
      intros [= <-]. cbn [indexed_map_opt].
      rewrite (His_int is_int eq_refl). unfold is_int_simple_of.
      rewrite Hother. reflexivity.
    + intros simples. cbn [add_variant_parts parts_get_tag].
      destruct (parts_get_tag parts) as [get_tag | ]; [ | discriminate ].
      destruct (Hget_tag get_tag eq_refl) as (-> & Hknown).
      destruct (deduce_get_tag_simple _ _) as [ | other_get_tag | ] eqn:Hother;
        [ discriminate | intros [= <-] | intros [= <-] ].
      * split.
        -- cbn [flat_map].
           replace (get_tag_simples_of (id, fields))
             with [(id, other_get_tag)]; [ reflexivity | ].
           unfold get_tag_simples_of, get_tag_simple_of. cbn [fst snd].
           rewrite Hother. reflexivity.
        -- intros x [<- | Hin]; [ | exact (Hknown x Hin) ].
           unfold get_tag_simple_of. cbn [snd]. rewrite Hother. discriminate.
      * split.
        -- cbn [flat_map].
           replace (get_tag_simples_of (id, fields))
             with (@nil (Index.t * simple)); [ reflexivity | ].
           unfold get_tag_simples_of, get_tag_simple_of. cbn [fst snd].
           rewrite Hother. reflexivity.
        -- intros x [<- | Hin]; [ | exact (Hknown x Hin) ].
           unfold get_tag_simple_of. cbn [snd]. rewrite Hother. discriminate.
Qed.

Lemma deduce_is_int_simple_cases (blocks : Or_unknown.t row_like_for_blocks)
    (imms : Or_unknown.t flambda_type) (is_int : option variable)
    (simple : simple) :
  deduce_is_int_simple blocks imms is_int = Some simple ->
  (∃ b : bool, simple = untagged_const_bool b)
  ∨ ∃ var : variable, is_int = Some var ∧ simple = Var var.
Proof.
  unfold deduce_is_int_simple.
  destruct (deduce_is_int blocks imms) as [b | ];
    [ intros [= <-]; left; exists b; reflexivity | ].
  destruct is_int as [var | ]; [ | discriminate ].
  intros [= <-]. right. exists var. split; reflexivity.
Qed.

Lemma deduce_get_tag_simple_cases (blocks : Or_unknown.t row_like_for_blocks)
    (get_tag : option variable) (simple : simple) :
  deduce_get_tag_simple blocks get_tag = Ok simple ->
  (∃ tag : Z, simple = untagged_const_int tag)
  ∨ ∃ var : variable, get_tag = Some var ∧ simple = Var var.
Proof.
  unfold deduce_get_tag_simple.
  destruct (deduce_get_tag blocks) as [ | tag | ];
    [ | intros [= <-]; left; exists tag; reflexivity | discriminate ].
  destruct get_tag as [var | ]; [ | discriminate ].
  intros [= <-]. right. exists var. split; reflexivity.
Qed.

(* The parts gathered of the first head and of those that follow it,
   where each has a part to join and an allocation mode. *)
Lemma gathered_parts {A : Type}
    (read : head_of_kind_value_non_null -> option (A * Alloc_mode.t))
    (first_id : Index.t) (first : A)
    (other_heads : list (Index.t * head_of_kind_value_non_null))
    (l : list (Index.t * (A * Alloc_mode.t))) :
  NoDup (first_id :: map fst other_heads) ->
  indexed_map_opt (fun _ head => read head) other_heads = Some l ->
  join_args_distinct
    (map (fun x => (fst x, fst (snd x))) l ++ [(first_id, first)])
  ∧ (∀ (index : Index.t) (head : head_of_kind_value_non_null),
       In (index, head) other_heads ->
       ∃ (a : A) (mode : Alloc_mode.t),
         read head = Some (a, mode)
         ∧ In (index, a)
             (map (fun x => (fst x, fst (snd x))) l ++ [(first_id, first)]))
  ∧ In (first_id, first)
      (map (fun x => (fst x, fst (snd x))) l ++ [(first_id, first)])
  ∧ ∀ (index : Index.t) (a : A),
      In (index, a)
        (map (fun x => (fst x, fst (snd x))) l ++ [(first_id, first)]) ->
      (index = first_id ∧ a = first)
      ∨ ∃ (head : head_of_kind_value_non_null) (mode : Alloc_mode.t),
          In (index, head) other_heads ∧ read head = Some (a, mode).
Proof.
  intros Hdistinct Hl.
  destruct (indexed_map_opt_spec _ _ l Hl) as (Hfst & Hto & Hfrom).
  split; [ | split; [ | split ] ].
  - unfold join_args_distinct. rewrite map_app, map_map. cbn [map fst].
    change (NoDup (map fst l ++ [first_id])). rewrite Hfst.
    apply snoc_distinct. exact Hdistinct.
  - intros index head Hin.
    destruct (Hto index head Hin) as ([a mode] & Hread & Hin').
    exists a, mode. split; [ exact Hread | ].
    apply in_or_app. left. apply in_map_iff. exists (index, (a, mode)).
    split; [ reflexivity | exact Hin' ].
  - apply in_or_app. right. left. reflexivity.
  - intros index a Hin. apply in_app_or in Hin as [Hin | [[= <- <-] | []]].
    + apply in_map_iff in Hin as ([index' [a' mode]] & [= <- <-] & Hin).
      destruct (Hfrom index' (a', mode) Hin) as (head & Hread & Hin').
      right. exists head, mode. split; [ exact Hin' | exact Hread ].
    + left. split; reflexivity.
Qed.

Lemma read_boxed_number_eq (k : boxable_number)
    (head : head_of_kind_value_non_null) (contents : flambda_type)
    (mode : Alloc_mode.t) :
  read_boxed_number k head = Some (contents, mode) ->
  head = Boxed_number k contents mode.
Proof.
  destruct head as [ | | k' contents' mode' | | ]; try discriminate. simpl.
  destruct (kind_eqb (unboxed_kind k) (unboxed_kind k')) eqn:Hkind;
    [ | discriminate ].
  intros [= <- <-]. apply kind_eqb_true in Hkind.
  destruct k, k'; try discriminate Hkind; reflexivity.
Qed.

Lemma read_closures_eq (head : head_of_kind_value_non_null)
    (closures : row_like_for_closures) (mode : Alloc_mode.t) :
  read_closures head = Some (closures, mode) -> head = Closures closures mode.
Proof. destruct head; try discriminate. intros [= <- <-]. reflexivity. Qed.

(* A variant, of its fields. *)
Definition variant_of (fields : variant_fields) : head_of_kind_value_non_null :=
  Variant (variant_is_int fields) (variant_immediates fields)
    (variant_get_tag fields) (variant_blocks fields)
    (variant_extensions_of fields) (variant_is_unique fields).

(* ======================================================================== *)
(* Fields                                                                   *)
(* ======================================================================== *)

Lemma free_variables_unknown (k : flambda_kind) :
  free_variables (unknown k) = [].
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma aliases_with_kinds_unknown (k : flambda_kind) :
  aliases_with_kinds (unknown k) = [].
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma free_variables_bottom (k : flambda_kind) :
  free_variables (bottom k) = [].
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma aliases_with_kinds_bottom (k : flambda_kind) :
  aliases_with_kinds (bottom k) = [].
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma free_variables_alias (k : flambda_kind) (s : simple) :
  free_variables (alias_type_of k s) = free_variables_of_simple s.
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma aliases_with_kinds_alias (k : flambda_kind) (s : simple) :
  aliases_with_kinds (alias_type_of k s) = [(s, k)].
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma kind_unknown (k : flambda_kind) : kind (unknown k) = k.
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma kind_bottom (k : flambda_kind) : kind (bottom k) = k.
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma kind_alias (k : flambda_kind) (s : simple) : kind (alias_type_of k s) = k.
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

(* The field of a block of a shape, at an index, is of the kind that
   the shape gives the index. *)
Lemma block_shape_element_kind_of_field (shape : block_shape) (tag : Z)
    (fields : list value) (index : nat) (field : value) (kind : flambda_kind) :
  block_has_shape shape tag fields ->
  nth_error fields index = Some field ->
  block_shape_element_kind shape index = Some kind ->
  kind_of_value field = kind.
Proof.
  intros [_ Hkinds] Hfield Hkind.
  apply (map_nth_error kind_of_value) in Hfield. rewrite Hkinds in Hfield.
  destruct shape as [[ | prefix suffix] | ]; simpl in Hkind, Hfield.
  - injection Hkind as <-. apply nth_error_In, repeat_spec in Hfield.
    exact Hfield.
  - destruct (Nat.ltb index prefix) eqn:Hlt.
    + injection Hkind as <-. apply Nat.ltb_lt in Hlt.
      rewrite nth_error_app1 in Hfield by (rewrite repeat_length; exact Hlt).
      apply nth_error_In, repeat_spec in Hfield. exact Hfield.
    + apply Nat.ltb_ge in Hlt.
      rewrite nth_error_app2 in Hfield by (rewrite repeat_length; exact Hlt).
      rewrite repeat_length, nth_error_map in Hfield. congruence.
  - injection Hkind as <-. apply nth_error_In, repeat_spec in Hfield.
    exact Hfield.
Qed.

(* The least of the lengths of arrays of fields. *)
Lemma least_length_le (others : list (Index.t * list flambda_type)) :
  ∀ init : nat,
    List.fold_left
      (fun least_length other_fields =>
         Nat.min least_length (length (snd other_fields)))
      others init
    <= init
    ∧ ∀ (index : Index.t) (fields : list flambda_type),
        In (index, fields) others ->
        List.fold_left
          (fun least_length other_fields =>
             Nat.min least_length (length (snd other_fields)))
          others init
        <= length fields.
Proof.
  induction others as [ | [id fields'] others IH]; intros init; simpl.
  - split; [ lia | intros ? ? [] ].
  - destruct (IH (Nat.min init (length fields'))) as (Hinit & Hothers).
    split; [ lia | ].
    intros index fields [[= <- <-] | Hin]; [ lia | exact (Hothers _ _ Hin) ].
Qed.

(* ======================================================================== *)
(* What [deduce_is_int] and [deduce_get_tag] answer                         *)
(* ======================================================================== *)

Lemma block_not_immediate (statics : static_env) (i tag : Z)
    (fields : list value) :
  block_of statics (Value_const (Tagged_immediate i)) <> Some (tag, fields).
Proof. discriminate. Qed.

Theorem deduce_is_int_simple_is_sound : deduce_is_int_simple_sound.
Proof.
  intros statics vars v is_int immediates get_tag blocks extensions is_unique
    simple Hsatisfied Hsimple.
  rewrite variant_satisfied in Hsatisfied.
  unfold deduce_is_int_simple, deduce_is_int in Hsimple.
  destruct Hsatisfied
    as [(i & -> & His_int & Himmediates & _)
       | (tag & fields & Hblock & His_int & _ & Hblocks & _)].
  - (* An immediate. *)
    left. split; [ exists i; reflexivity | ].
    destruct blocks as [blocks | ].
    + destruct (row_like_for_blocks_is_bottom blocks).
      * injection Hsimple as <-. reflexivity.
      * destruct immediates as [immediates | ].
        -- destruct (is_obviously_bottom immediates) eqn:Hbottom.
           ++ exfalso.
              exact (is_obviously_bottom_satisfied _ _ _ _ Hbottom Himmediates).
           ++ destruct is_int as [is_int | ]; [ | discriminate ].
              injection Hsimple as <-. exact His_int.
        -- destruct is_int as [is_int | ]; [ | discriminate ].
           injection Hsimple as <-. exact His_int.
    + destruct immediates as [immediates | ].
      * destruct (is_obviously_bottom immediates) eqn:Hbottom.
        -- exfalso.
           exact (is_obviously_bottom_satisfied _ _ _ _ Hbottom Himmediates).
        -- destruct is_int as [is_int | ]; [ | discriminate ].
           injection Hsimple as <-. exact His_int.
      * destruct is_int as [is_int | ]; [ | discriminate ].
        injection Hsimple as <-. exact His_int.
  - (* A block. *)
    right. split; [ exists tag, fields; exact Hblock | ].
    destruct blocks as [blocks | ].
    + destruct (row_like_for_blocks_is_bottom blocks) eqn:Hbottom.
      * exfalso.
        exact (row_like_for_blocks_is_bottom_satisfied _ _ _ _ _ Hbottom
                 Hblocks).
      * destruct immediates as [immediates | ].
        -- destruct (is_obviously_bottom immediates).
           ++ injection Hsimple as <-. reflexivity.
           ++ destruct is_int as [is_int | ]; [ | discriminate ].
              injection Hsimple as <-. exact His_int.
        -- destruct is_int as [is_int | ]; [ | discriminate ].
           injection Hsimple as <-. exact His_int.
    + destruct immediates as [immediates | ].
      * destruct (is_obviously_bottom immediates).
        -- injection Hsimple as <-. reflexivity.
        -- destruct is_int as [is_int | ]; [ | discriminate ].
           injection Hsimple as <-. exact His_int.
      * destruct is_int as [is_int | ]; [ | discriminate ].
        injection Hsimple as <-. exact His_int.
Qed.

(* The tag of a block that a row-like structure with no other case
   admits is one of its known tags. *)
Lemma row_like_tag_among (statics : static_env) (vars : env) (tag : Z)
    (fields : list value)
    (known_tags : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (alloc_mode : Alloc_mode.t) :
  row_like_for_blocks_satisfied statics vars tag fields
    (Row_like_for_blocks known_tags Or_bottom.Bottom alloc_mode) ->
  Keyed.among Z.eqb tag (Keyed.domain Z.eqb known_tags) = true.
Proof.
  rewrite row_like_for_blocks_find_iff,
    (keyed_among_domain Z Z.eqb Z.eqb_eq).
  unfold Keyed.is_bound.
  destruct (Keyed.lookup Z.eqb tag known_tags); [ reflexivity | intros [] ].
Qed.

Theorem deduce_get_tag_simple_is_sound : deduce_get_tag_simple_sound.
Proof.
  intros statics vars v is_int immediates get_tag blocks extensions is_unique
    tag fields Hsatisfied Hblock.
  rewrite variant_satisfied in Hsatisfied.
  destruct Hsatisfied
    as [(i & -> & _) | (tag' & fields' & Hblock' & _ & Hget_tag & Hblocks & _)];
    [ exfalso; exact (block_not_immediate _ _ _ _ Hblock) | ].
  rewrite Hblock in Hblock'. injection Hblock' as <- <-.
  assert (Hvar :
    match
      match get_tag with
      | Some var => Ok (Var var)
      | None => @Unknown simple
      end
    with
    | Bottom => False
    | Ok simple =>
      eval_simple vars simple = Some (Value_const (Naked_immediate tag))
    | Unknown => True
    end).
  { destruct get_tag as [var | ]; [ exact Hget_tag | exact I ]. }
  unfold deduce_get_tag_simple, deduce_get_tag.
  destruct blocks as [[known_tags [other | ] alloc_mode] | ];
    [ exact Hvar | | exact Hvar ].
  unfold row_like_for_blocks_all_tags. cbv beta iota.
  pose proof (row_like_tag_among _ _ _ _ _ _ Hblocks) as Hamong.
  unfold row_like_block_case in *.
  destruct (Keyed.is_empty (Keyed.domain Z.eqb known_tags)) eqn:Hempty.
  - apply (keyed_is_empty_among Z Z.eqb Z.eqb_eq) with (key := tag) in Hempty.
    exact (eq_true_false_abs _ Hamong Hempty).
  - destruct (Keyed.get_singleton Z.eqb (Keyed.domain Z.eqb known_tags))
      as [tag' | ] eqn:Hsingleton; [ | exact Hvar ].
    pose proof
      (proj1 (keyed_get_singleton Z Z.eqb Z.eqb_eq tag' _) Hsingleton tag)
      as Heq.
    rewrite Heq in Hamong. apply Z.eqb_eq in Hamong. subst tag'.
    reflexivity.
Qed.

(* ======================================================================== *)
(* The lists at which types are kinded                                      *)
(* ======================================================================== *)

(* The functions that call the join of types are proved sound once,
   for a list of [Simple]s with kinds at which the things given and
   the answers are kinded. The list is the aliases of a type
   ([aliases_with_kinds]), to which two things can be added.
   - With [with_equations], the names that have equations inside the
     type, each at the kind of the type of its equation
     ([equation_names_with_kinds]).
   - With [with_checks], for each test of the well-formedness of types
     that the type fails, a [Simple] at a kind of which it is not
     ([ill_formed]). A type is kinded at such a list, in an environment
     or in the target, only if it is well formed.
   The lists are defined as [aliases_with_kinds] is, and are
   [aliases_with_kinds] by computation where nothing is added. *)

(* A constant at a kind that is not its own. *)
Definition ill_formed : simple * flambda_kind :=
  (Const (Naked_immediate 0%Z), Value).

(* The tests of the well-formedness of types that are about one
   structure, and not about the types inside it: the cases of blocks
   (conditions 6, 7 and 9), an entry of closures under an index and
   the cases of closures (conditions 5, 8 and 9). *)
Definition block_case_shape_ok (case : row_like_block_case) : bool :=
  block_product_fits (case_index case) (case_maps_to case)
  && block_shape_canonical_b (index_shape (case_index case)).

Definition blocks_shape_ok
    (known_tags : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (other_tags : Or_bottom.t row_like_block_case) : bool :=
  Keyed.distinct Z.eqb (keys known_tags)
  && Keyed.for_all_bindings_b Z.eqb
       (fun _ case =>
          match case with
          | Or_unknown.Known case => block_case_shape_ok case
          | Or_unknown.Unknown => true
          end)
       known_tags
  && match other_tags with
     | Or_bottom.Ok case => block_case_shape_ok case
     | Or_bottom.Bottom => true
     end.

Definition closures_entry_shape_ok
    (index : row_like_index set_of_closures_contents unit)
    (entry : closures_entry) : bool :=
  match entry with
  | Closures_entry function_types closure_types value_slot_types =>
    closures_entry_slots_fit index (domain function_types)
      (domain closure_types) (domain value_slot_types)
    && distinct (keys function_types)
    && distinct (keys closure_types)
    && distinct (keys value_slot_types)
    && for_all_bindings_b (fun _ ty => kind_eqb (kind ty) Value) closure_types
  end.

Definition closures_shape_ok
    (known_closures : finite_map row_like_closures_case) : bool :=
  distinct (keys known_closures)
  && for_all_bindings_b
       (fun slot case =>
          closures_index_has_slot slot (case_index case)
          && closures_entry_shape_ok (case_index case) (case_maps_to case))
       known_closures.

Section Kinds.

Variable with_equations : bool.
Variable with_checks : bool.

Definition kinds_of_equation (n : name) (ty : flambda_type)
    : list (simple * flambda_kind) :=
  if with_equations then [(simple_of_name n, kind ty)] else [].

Definition check (test : bool) : list (simple * flambda_kind) :=
  if with_checks then if test then [] else [ill_formed] else [].

Fixpoint kinds_with (t : flambda_type) : list (simple * flambda_kind) :=
  match t with
  | Type_value ty => aliases_of_descr Value kinds_of_head_of_kind_value ty
  | Type_naked_immediate ty =>
    check (descr_well_formed head_of_kind_naked_immediate_well_formed ty)
    ++ aliases_of_descr (Naked_number Kind_naked_immediate) (fun _ => []) ty
  | Type_naked_number k ty =>
    check (has_naked_number_types k && descr_well_formed non_empty ty)
    ++ aliases_of_descr (Naked_number k) (fun _ => []) ty
  | Type_rec_info ty => aliases_of_descr Rec_info (fun _ => []) ty
  | Type_region ty => aliases_of_descr Region (fun _ => []) ty
  end

with kinds_of_head_of_kind_value (head : head_of_kind_value)
    : list (simple * flambda_kind) :=
  match head with
  | Head_of_kind_value non_null is_null =>
    match non_null with
    | Ok head => kinds_of_head_of_kind_value_non_null head
    | Unknown | Bottom => []
    end
    ++ match is_null with
       | Not_null => []
       | Maybe_null is_null => aliases_of_relation is_null
       end
  end

with kinds_of_head_of_kind_value_non_null
    (head : head_of_kind_value_non_null) : list (simple * flambda_kind) :=
  match head with
  | Variant is_int immediates get_tag blocks extensions _ =>
    aliases_of_relation is_int
    ++ match immediates with
       | Or_unknown.Known ty =>
         check (kind_eqb (kind ty) (Naked_number Kind_naked_immediate))
         ++ kinds_with ty
       | Or_unknown.Unknown => []
       end
    ++ aliases_of_relation get_tag
    ++ match blocks with
       | Or_unknown.Known blocks => kinds_of_row_like_for_blocks blocks
       | Or_unknown.Unknown => []
       end
    ++ match extensions with
       | No_extensions => []
       | Ext when_immediate when_block =>
         kinds_of_env_extension when_immediate
         ++ kinds_of_env_extension when_block
       end
  | Mutable_block _ => []
  | String strs => check (non_empty strs)
  | Boxed_number k contents _ =>
    check (kind_eqb (kind contents) (unboxed_kind k)) ++ kinds_with contents
  | Closures by_function_slot _ =>
    kinds_of_row_like_for_closures by_function_slot
  end

with kinds_of_row_like_for_blocks (blocks : row_like_for_blocks)
    : list (simple * flambda_kind) :=
  match blocks with
  | Row_like_for_blocks known_tags other_tags _ =>
    check (blocks_shape_ok known_tags other_tags)
    ++ flat_map
         (fun case =>
            match snd case with
            | Or_unknown.Known case =>
              aliases_of_block_case kinds_with kinds_of_env_extension case
            | Or_unknown.Unknown => []
            end)
         known_tags
    ++ match other_tags with
       | Or_bottom.Ok case =>
         aliases_of_block_case kinds_with kinds_of_env_extension case
       | Or_bottom.Bottom => []
       end
  end

with kinds_of_row_like_for_closures (closures : row_like_for_closures)
    : list (simple * flambda_kind) :=
  match closures with
  | Row_like_for_closures known_closures =>
    check (closures_shape_ok known_closures)
    ++ flat_map
         (fun case =>
            kinds_of_closures_entry (case_maps_to (snd case))
            ++ kinds_of_env_extension (case_env_extension (snd case)))
         known_closures
  end

with kinds_of_closures_entry (entry : closures_entry)
    : list (simple * flambda_kind) :=
  match entry with
  | Closures_entry _ closure_types value_slot_types =>
    flat_map (fun binding => kinds_with (snd binding)) closure_types
    ++ flat_map (fun binding => kinds_with (snd binding)) value_slot_types
  end

with kinds_of_env_extension (extension : env_extension)
    : list (simple * flambda_kind) :=
  match extension with
  | Env_extension equations =>
    check (Keyed.distinct name_eqb (keys equations))
    ++ flat_map
         (fun binding =>
            kinds_of_equation (fst binding) (snd binding)
            ++ kinds_with (snd binding))
         equations
  end.

(* A type in a place that asks for a kind (the immediates of a
   variant, the contents of a boxed number). *)
Definition kinds_at (k : flambda_kind) (ty : flambda_type)
    : list (simple * flambda_kind) :=
  check (kind_eqb (kind ty) k) ++ kinds_with ty.

End Kinds.

(* A name with a kind, as a [Simple] with a kind. *)
Definition name_at_kind (binding : name * flambda_kind)
    : simple * flambda_kind :=
  (simple_of_name (fst binding), snd binding).

(* With nothing added, the lists are the aliases. *)
Lemma kinds_alone :
  (∀ t : flambda_type, kinds_with false false t = aliases_with_kinds t)
  ∧ (∀ head : head_of_kind_value,
       kinds_of_head_of_kind_value false false head
       = aliases_of_head_of_kind_value head)
  ∧ (∀ head : head_of_kind_value_non_null,
       kinds_of_head_of_kind_value_non_null false false head
       = aliases_of_head_of_kind_value_non_null head)
  ∧ (∀ blocks : row_like_for_blocks,
       kinds_of_row_like_for_blocks false false blocks
       = aliases_of_row_like_for_blocks blocks)
  ∧ (∀ closures : row_like_for_closures,
       kinds_of_row_like_for_closures false false closures
       = aliases_of_row_like_for_closures closures)
  ∧ (∀ entry : closures_entry,
       kinds_of_closures_entry false false entry
       = aliases_of_closures_entry entry)
  ∧ (∀ extension : env_extension,
       kinds_of_env_extension false false extension
       = aliases_of_env_extension extension).
Proof. repeat split; reflexivity. Qed.

(* Membership in two lists made by [flat_map] over one list. *)
Lemma in_flat_map_split {A B C : Type} (f : A -> list B) (g : A -> list B)
    (h : A -> list C) (m : C -> B) (l : list A) (b : B) :
  (∀ a : A, a ∈ l -> b ∈ f a <-> b ∈ g a ∨ b ∈ map m (h a)) ->
  b ∈ flat_map f l <-> b ∈ flat_map g l ∨ b ∈ map m (flat_map h l).
Proof.
  induction l as [ | a l IH]; intros H; simpl; [ tauto | ].
  rewrite map_app, !in_app_iff, (H a (or_introl eq_refl)), IH; [ tauto | ].
  intros a' Hin. apply H. right. exact Hin.
Qed.

Lemma or_iff_split (A A1 A2 B B1 B2 : Prop) :
  (A <-> A1 ∨ A2) -> (B <-> B1 ∨ B2) -> (A ∨ B <-> (A1 ∨ B1) ∨ A2 ∨ B2).
Proof. tauto. Qed.

(* With them, they are the aliases and the names of the equations. *)
Lemma kinds_with_equations (p : simple * flambda_kind) :
  (∀ t : flambda_type,
     p ∈ kinds_with true false t
     <-> p ∈ aliases_with_kinds t
         ∨ p ∈ map name_at_kind (equation_names_with_kinds t))
  ∧ (∀ head : head_of_kind_value,
       p ∈ kinds_of_head_of_kind_value true false head
       <-> p ∈ aliases_of_head_of_kind_value head
           ∨ p ∈ map name_at_kind (equation_names_of_head_of_kind_value head))
  ∧ (∀ head : head_of_kind_value_non_null,
       p ∈ kinds_of_head_of_kind_value_non_null true false head
       <-> p ∈ aliases_of_head_of_kind_value_non_null head
           ∨ p ∈ map name_at_kind
                   (equation_names_of_head_of_kind_value_non_null head))
  ∧ (∀ blocks : row_like_for_blocks,
       p ∈ kinds_of_row_like_for_blocks true false blocks
       <-> p ∈ aliases_of_row_like_for_blocks blocks
           ∨ p ∈ map name_at_kind
                   (equation_names_of_row_like_for_blocks blocks))
  ∧ (∀ closures : row_like_for_closures,
       p ∈ kinds_of_row_like_for_closures true false closures
       <-> p ∈ aliases_of_row_like_for_closures closures
           ∨ p ∈ map name_at_kind
                   (equation_names_of_row_like_for_closures closures))
  ∧ (∀ entry : closures_entry,
       p ∈ kinds_of_closures_entry true false entry
       <-> p ∈ aliases_of_closures_entry entry
           ∨ p ∈ map name_at_kind (equation_names_of_closures_entry entry))
  ∧ (∀ extension : env_extension,
       p ∈ kinds_of_env_extension true false extension
       <-> p ∈ aliases_of_env_extension extension
           ∨ p ∈ map name_at_kind
                   (equation_names_of_env_extension extension)).
Proof.
  assert (Hcase :
    ∀ case : row_like_block_case,
      P_block_case
        (fun t =>
           p ∈ kinds_with true false t
           <-> p ∈ aliases_with_kinds t
               ∨ p ∈ map name_at_kind (equation_names_with_kinds t))
        (fun extension =>
           p ∈ kinds_of_env_extension true false extension
           <-> p ∈ aliases_of_env_extension extension
               ∨ p ∈ map name_at_kind
                       (equation_names_of_env_extension extension))
        case ->
      p
      ∈ aliases_of_block_case (kinds_with true false)
          (kinds_of_env_extension true false) case
      <-> p ∈ aliases_of_block_case aliases_with_kinds
                aliases_of_env_extension case
          ∨ p ∈ map name_at_kind
                  (equation_names_of_block_case equation_names_with_kinds
                     equation_names_of_env_extension case)).
  { intros case [Htypes Hextension].
    unfold aliases_of_block_case, equation_names_of_block_case.
    rewrite map_app, !in_app_iff, Hextension,
      (in_flat_map_split _ _ _ name_at_kind _ p Htypes).
    tauto. }
  apply types_induction.
  - intros [ | [head | s] | ] H; simpl; tauto.
  - intros [ | [head | s] | ]; simpl; tauto.
  - intros k [ | [head | s] | ]; simpl; tauto.
  - intros [ | [head | s] | ]; simpl; tauto.
  - intros [ | [head | s] | ]; simpl; tauto.
  - intros [ | head | ] is_null H; simpl; rewrite ?in_app_iff, ?H; tauto.
  - intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions. simpl.
    destruct immediates as [ty | ], blocks as [blocks | ],
      extensions as [ | when_immediate when_block ];
      simpl; rewrite ?map_app, ?in_app_iff, ?Himmediates, ?Hblocks,
        ?(proj1 Hextensions), ?(proj2 Hextensions);
      simpl; tauto.
  - intros alloc_mode. simpl. tauto.
  - intros k contents alloc_mode H. exact H.
  - intros by_function_slot alloc_mode H. exact H.
  - intros strs. simpl. tauto.
  - intros known_tags other_tags alloc_mode Hknown Hother. simpl.
    rewrite map_app, !in_app_iff. apply or_iff_split.
    + apply in_flat_map_split. intros [tag [case | ]] Hin; simpl; [ | tauto ].
      exact (Hcase case (Hknown tag case Hin)).
    + destruct other_tags as [case | ]; [ exact (Hcase case Hother) | ].
      simpl. tauto.
  - intros known_closures Hcases. simpl. apply in_flat_map_split.
    intros [slot case] Hin. destruct (Hcases slot case Hin) as [Hentry Hext].
    simpl. rewrite map_app, !in_app_iff, Hentry, Hext. tauto.
  - intros function_types closure_types value_slot_types Hclosures Hslots.
    simpl. rewrite map_app, !in_app_iff.
    apply or_iff_split; apply in_flat_map_split; intros [key ty] Hin.
    + exact (Hclosures key ty Hin).
    + exact (Hslots key ty Hin).
  - intros equations Htypes. simpl. apply in_flat_map_split.
    intros [n ty] Hin. simpl. rewrite (Htypes n ty Hin).
    unfold name_at_kind. simpl. tauto.
Qed.

(* A list with the equations is kinded, in an environment or in the
   target, when the aliases and the names of the equations are. *)
Lemma kinds_all {X : Type} (kinds aliases : X -> list (simple * flambda_kind))
    (names : X -> list (name * flambda_kind))
    (P : simple -> flambda_kind -> Prop) (x : X) :
  (∀ p : simple * flambda_kind,
     p ∈ kinds x <-> p ∈ aliases x ∨ p ∈ map name_at_kind (names x)) ->
  (∀ (simple : simple) (kind : flambda_kind),
     (simple, kind) ∈ kinds x -> P simple kind)
  <-> (∀ (simple : simple) (kind : flambda_kind),
         (simple, kind) ∈ aliases x -> P simple kind)
      ∧ ∀ (n : name) (kind : flambda_kind),
          (n, kind) ∈ names x -> P (simple_of_name n) kind.
Proof.
  intros Hiff. split.
  - intros H. split.
    + intros simple kind Hin. apply H, Hiff. left. exact Hin.
    + intros n kind Hin. apply H, Hiff. right. apply in_map_iff.
      exists (n, kind). split; [ reflexivity | exact Hin ].
  - intros [Haliases Hnames] simple kind Hin.
    apply Hiff in Hin as [Hin | Hin]; [ exact (Haliases simple kind Hin) | ].
    apply in_map_iff in Hin as ([n kind'] & [= <- <-] & Hin).
    exact (Hnames n kind' Hin).
Qed.

Lemma kinds_with_equations_all (P : simple -> flambda_kind -> Prop)
    (t : flambda_type) :
  (∀ (simple : simple) (kind : flambda_kind),
     (simple, kind) ∈ kinds_with true false t -> P simple kind)
  <-> (∀ (simple : simple) (kind : flambda_kind),
         (simple, kind) ∈ aliases_with_kinds t -> P simple kind)
      ∧ ∀ (n : name) (kind : flambda_kind),
          (n, kind) ∈ equation_names_with_kinds t ->
          P (simple_of_name n) kind.
Proof.
  apply kinds_all. intros p. exact (proj1 (kinds_with_equations p) t).
Qed.

Lemma kinds_of_extension_all (P : simple -> flambda_kind -> Prop)
    (extension : env_extension) :
  (∀ (simple : simple) (kind : flambda_kind),
     (simple, kind) ∈ kinds_of_env_extension true false extension ->
     P simple kind)
  <-> (∀ (simple : simple) (kind : flambda_kind),
         (simple, kind) ∈ aliases_of_env_extension extension ->
         P simple kind)
      ∧ ∀ (n : name) (kind : flambda_kind),
          (n, kind) ∈ equation_names_of_env_extension extension ->
          P (simple_of_name n) kind.
Proof.
  apply kinds_all. intros p.
  exact (proj2 (proj2 (proj2 (proj2 (proj2 (proj2 (kinds_with_equations p))))))
           extension).
Qed.

(* ======================================================================== *)
(* A type that is kinded at a list with the checks is well formed           *)
(* ======================================================================== *)

Arguments check : simpl never.

Lemma check_off (test : bool) : check false test = [].
Proof. reflexivity. Qed.

Lemma lookup_of_distinct_in {K A : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) (bindings : list (K * A)) :
  Keyed.distinct eqb (map fst bindings) = true ->
  ∀ (key : K) (a : A),
    (key, a) ∈ bindings -> Keyed.lookup eqb key bindings = Some a.
Proof.
  induction bindings as [ | [bound b] rest IH]; intros Hdistinct key a Hin;
    [ destruct Hin | ].
  cbn in Hdistinct. apply andb_true_iff in Hdistinct as [Hfresh Hdistinct].
  rewrite keyed_lookup_cons. destruct Hin as [[= <- <-] | Hin].
  - rewrite (keyed_eqb_refl eqb Heqb). reflexivity.
  - destruct (eqb key bound) eqn:Hkey; [ | exact (IH Hdistinct key a Hin) ].
    exfalso. apply Heqb in Hkey. subst bound.
    apply negb_true_iff in Hfresh.
    assert (Hamong : Keyed.among eqb key (map fst rest) = true).
    { apply (keyed_among_in K eqb Heqb). apply in_map_iff.
      exists (key, a). split; [ reflexivity | exact Hin ]. }
    congruence.
Qed.

(* A test of the bindings of a map that binds a key once is a test of
   the pairs of its list. *)
Lemma for_all_bindings_b_in {K A : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) (p : K -> A -> bool)
    (bindings : Keyed.finite_map K A) :
  Keyed.distinct eqb (keys bindings) = true ->
  Keyed.for_all_bindings_b eqb p bindings = true
  <-> ∀ (key : K) (a : A), (key, a) ∈ bindings -> p key a = true.
Proof.
  intros Hdistinct.
  rewrite (keyed_for_all_bindings_b K eqb Heqb),
    (keyed_for_all_bindings_lookup K eqb Heqb).
  split; intros H key a Hin.
  - exact (H key a
             (lookup_of_distinct_in eqb Heqb bindings Hdistinct key a Hin)).
  - exact (H key a (keyed_lookup_in eqb Heqb key a bindings Hin)).
Qed.

(* The types of an entry of closures are well formed. *)
Definition entry_types_well_formed (entry : closures_entry) : Prop :=
  match entry with
  | Closures_entry _ closure_types value_slot_types =>
    (∀ (g : function_slot) (ty : flambda_type),
       (g, ty) ∈ closure_types -> type_well_formed ty = true)
    ∧ ∀ (w : value_slot) (ty : flambda_type),
        (w, ty) ∈ value_slot_types -> type_well_formed ty = true
  end.

Section Checked.

(* What it is to be of a kind, in an environment or in the target:
   [ill_formed] is not. *)
Variable P : simple -> flambda_kind -> Prop.

Hypothesis Hill : ¬ P (fst ill_formed) (snd ill_formed).

Definition all_at (l : list (simple * flambda_kind)) : Prop :=
  ∀ (simple : simple) (kind : flambda_kind),
    (simple, kind) ∈ l -> P simple kind.

Lemma all_at_app (l l' : list (simple * flambda_kind)) :
  all_at (l ++ l') <-> all_at l ∧ all_at l'.
Proof.
  unfold all_at. split.
  - intros H. split; intros s k Hin; apply H, in_or_app; [ left | right ];
      exact Hin.
  - intros [H H'] s k Hin. apply in_app_or in Hin as [Hin | Hin]; auto.
Qed.

Lemma all_at_nil : all_at [] <-> True.
Proof. split; [ intros _; exact I | intros _ s k [] ]. Qed.

Lemma all_at_check (test : bool) : all_at (check true test) <-> test = true.
Proof.
  unfold check. destruct test; [ rewrite all_at_nil; tauto | ].
  split; [ | discriminate ]. intros H. exfalso. apply Hill.
  apply H. left. reflexivity.
Qed.

Lemma all_at_flat_map {A : Type} (f : A -> list (simple * flambda_kind))
    (l : list A) :
  all_at (flat_map f l) <-> ∀ a : A, a ∈ l -> all_at (f a).
Proof.
  unfold all_at. split.
  - intros H a Ha s k Hin. apply H, in_flat_map. exists a. split; assumption.
  - intros H s k Hin. apply in_flat_map in Hin as (a & Ha & Hin).
    exact (H a Ha s k Hin).
Qed.

Variable e : bool.

Lemma kinds_checked :
  (∀ t : flambda_type,
     all_at (kinds_with e true t)
     <-> all_at (kinds_with e false t) ∧ type_well_formed t = true)
  ∧ (∀ head : head_of_kind_value,
       all_at (kinds_of_head_of_kind_value e true head)
       <-> all_at (kinds_of_head_of_kind_value e false head)
           ∧ head_of_kind_value_well_formed head = true)
  ∧ (∀ head : head_of_kind_value_non_null,
       all_at (kinds_of_head_of_kind_value_non_null e true head)
       <-> all_at (kinds_of_head_of_kind_value_non_null e false head)
           ∧ head_of_kind_value_non_null_well_formed head = true)
  ∧ (∀ blocks : row_like_for_blocks,
       all_at (kinds_of_row_like_for_blocks e true blocks)
       <-> all_at (kinds_of_row_like_for_blocks e false blocks)
           ∧ row_like_for_blocks_well_formed blocks = true)
  ∧ (∀ closures : row_like_for_closures,
       all_at (kinds_of_row_like_for_closures e true closures)
       <-> all_at (kinds_of_row_like_for_closures e false closures)
           ∧ row_like_for_closures_well_formed closures = true)
  ∧ (∀ entry : closures_entry,
       all_at (kinds_of_closures_entry e true entry)
       <-> all_at (kinds_of_closures_entry e false entry)
           ∧ entry_types_well_formed entry)
  ∧ (∀ extension : env_extension,
       all_at (kinds_of_env_extension e true extension)
       <-> all_at (kinds_of_env_extension e false extension)
           ∧ env_extension_well_formed extension = true).
Proof.
  assert (Hcase :
    ∀ case : row_like_block_case,
      P_block_case
        (fun t =>
           all_at (kinds_with e true t)
           <-> all_at (kinds_with e false t) ∧ type_well_formed t = true)
        (fun extension =>
           all_at (kinds_of_env_extension e true extension)
           <-> all_at (kinds_of_env_extension e false extension)
               ∧ env_extension_well_formed extension = true)
        case ->
      all_at
        (aliases_of_block_case (kinds_with e true)
           (kinds_of_env_extension e true) case)
      <-> all_at
            (aliases_of_block_case (kinds_with e false)
               (kinds_of_env_extension e false) case)
          ∧ (∀ ty : flambda_type,
               ty ∈ case_maps_to case -> type_well_formed ty = true)
          ∧ env_extension_well_formed (case_env_extension case) = true).
  { intros case [Htypes Hextension]. unfold aliases_of_block_case.
    rewrite !all_at_app, !all_at_flat_map, Hextension. split.
    - intros (Htys & Hext & Hwf). split; [ | split ].
      + split; [ | exact Hext ].
        intros ty Hin. exact (proj1 (proj1 (Htypes ty Hin) (Htys ty Hin))).
      + intros ty Hin. exact (proj2 (proj1 (Htypes ty Hin) (Htys ty Hin))).
      + exact Hwf.
    - intros ((Htys & Hext) & Hwf_tys & Hwf). split; [ | split; assumption ].
      intros ty Hin.
      exact (proj2 (Htypes ty Hin) (conj (Htys ty Hin) (Hwf_tys ty Hin))). }
  apply types_induction.
  - intros [ | [head | s] | ] H; simpl; [ | exact H | | ];
      intuition (auto; congruence).
  - intros ty. simpl.
    rewrite ?all_at_app, ?all_at_check, ?check_off, ?all_at_nil. tauto.
  - intros k ty. simpl.
    rewrite ?all_at_app, ?all_at_check, ?check_off, ?all_at_nil. tauto.
  - intros [ | [head | s] | ]; simpl; intuition (auto; congruence).
  - intros [ | [head | s] | ]; simpl; intuition (auto; congruence).
  - intros [ | head | ] is_null H; simpl; rewrite ?all_at_app, ?H;
      intuition (auto; congruence).
  - intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions. simpl.
    destruct immediates as [ty | ], blocks as [blocks | ],
      extensions as [ | when_immediate when_block ];
      simpl;
      rewrite ?all_at_app, ?all_at_check, ?check_off, ?all_at_nil,
        ?andb_true_iff, ?Himmediates, ?Hblocks, ?(proj1 Hextensions),
        ?(proj2 Hextensions);
      intuition (auto; congruence).
  - intros alloc_mode. simpl. intuition (auto; congruence).
  - intros k contents alloc_mode H. simpl.
    rewrite ?all_at_app, ?all_at_check, ?check_off, ?all_at_nil,
      andb_true_iff, H.
    tauto.
  - intros by_function_slot alloc_mode H. exact H.
  - intros strs. simpl.
    rewrite ?all_at_check, ?check_off, ?all_at_nil. tauto.
  - (* The cases of blocks. *)
    intros known_tags other_tags alloc_mode Hknown Hother. simpl.
    rewrite ?all_at_app, ?all_at_check, ?check_off, ?all_at_nil,
      ?all_at_flat_map.
    split.
    + intros (Hshape & Hcases & Hother').
      unfold blocks_shape_ok in Hshape.
      apply andb_true_iff in Hshape as [Hshape Hother_shape].
      apply andb_true_iff in Hshape as [Hdistinct Hshapes].
      assert (Hshapes' :=
        proj1 (for_all_bindings_b_in Z.eqb z_eqb_decides_equality _
                 known_tags Hdistinct)
          Hshapes).
      clear Hshapes. rename Hshapes' into Hshapes.
      split.
      * split.
        -- intros [tag [case | ]] Hin; [ | apply all_at_nil; exact I ].
           exact (proj1 (proj1 (Hcase case (Hknown tag case Hin))
                           (Hcases _ Hin))).
        -- destruct other_tags as [case | ]; [ | exact Hother' ].
           exact (proj1 (proj1 (Hcase case Hother) Hother')).
      * apply (proj2 (row_like_for_blocks_well_formed_iff known_tags
                        other_tags alloc_mode)).
        split; [ exact Hdistinct | ]. split.
        -- intros tag case Hlookup.
           assert (Hin :=
             keyed_lookup_in Z.eqb z_eqb_decides_equality tag _ known_tags
               Hlookup).
           apply (proj2 (row_like_block_case_well_formed_iff case)).
           specialize (Hshapes tag _ Hin).
           unfold block_case_shape_ok in Hshapes.
           apply andb_true_iff in Hshapes as [Hfits Hcanonical].
           destruct (proj1 (Hcase case (Hknown tag case Hin)) (Hcases _ Hin))
             as (_ & Htys & Hext).
           split; [ exact Hfits | ].
           split; [ apply block_shape_canonical_b_iff; exact Hcanonical | ].
           split; assumption.
        -- destruct other_tags as [case | ]; [ | exact I ].
           apply (proj2 (row_like_block_case_well_formed_iff case)).
           unfold block_case_shape_ok in Hother_shape.
           apply andb_true_iff in Hother_shape as [Hfits Hcanonical].
           destruct (proj1 (Hcase case Hother) Hother') as (_ & Htys & Hext).
           split; [ exact Hfits | ].
           split; [ apply block_shape_canonical_b_iff; exact Hcanonical | ].
           split; assumption.
    + intros ((Hcases & Hother') & Hwf).
      destruct (proj1 (row_like_for_blocks_well_formed_iff known_tags
                         other_tags alloc_mode)
                  Hwf)
        as (Hdistinct & Hwf_cases & Hwf_other).
      assert (Hshape_of :
        ∀ case : row_like_block_case,
          row_like_block_case_well_formed case = true ->
          block_case_shape_ok case = true).
      { intros case Hwf_case.
        destruct (proj1 (row_like_block_case_well_formed_iff case) Hwf_case)
          as (Hfits & Hcanonical & _).
        unfold block_case_shape_ok. rewrite Hfits.
        apply block_shape_canonical_b_iff in Hcanonical. rewrite Hcanonical.
        reflexivity. }
      assert (Hin_wf :
        ∀ (tag : Z) (case : row_like_block_case),
          (tag, Or_unknown.Known case) ∈ known_tags ->
          row_like_block_case_well_formed case = true).
      { intros tag case Hin. apply (Hwf_cases tag).
        exact (lookup_of_distinct_in Z.eqb z_eqb_decides_equality known_tags
                 Hdistinct tag _ Hin). }
      split; [ | split ].
      * unfold blocks_shape_ok. rewrite !andb_true_iff.
        split; [ split; [ exact Hdistinct | ] | ].
        -- apply (proj2 (for_all_bindings_b_in Z.eqb z_eqb_decides_equality _
                            known_tags Hdistinct)).
           intros tag [case | ] Hin; [ | reflexivity ].
           exact (Hshape_of case (Hin_wf tag case Hin)).
        -- destruct other_tags as [case | ]; [ | reflexivity ].
           exact (Hshape_of case Hwf_other).
      * intros [tag [case | ]] Hin; [ | apply all_at_nil; exact I ].
        apply (Hcase case (Hknown tag case Hin)).
        split; [ exact (Hcases _ Hin) | ].
        destruct (proj1 (row_like_block_case_well_formed_iff case)
                    (Hin_wf tag case Hin))
          as (_ & _ & Htys & Hext).
        split; assumption.
      * destruct other_tags as [case | ]; [ | exact Hother' ].
        apply (Hcase case Hother). split; [ exact Hother' | ].
        destruct (proj1 (row_like_block_case_well_formed_iff case) Hwf_other)
          as (_ & _ & Htys & Hext).
        split; assumption.
  - (* The cases of closures. *)
    intros known_closures Hcases. simpl.
    rewrite ?all_at_app, ?all_at_check, ?check_off, ?all_at_nil,
      ?all_at_flat_map.
    assert (Hcase_iff :
      ∀ (slot : function_slot) (case : row_like_closures_case),
        row_like_closures_case_well_formed slot case = true
        <-> closures_index_has_slot slot (case_index case)
            && closures_entry_shape_ok (case_index case) (case_maps_to case)
            = true
            ∧ entry_types_well_formed (case_maps_to case)
            ∧ env_extension_well_formed (case_env_extension case) = true).
    { intros slot [[function_types closure_types value_slot_types] index
                     extension].
      unfold row_like_closures_case_well_formed.
      cbn [case_index case_maps_to case_env_extension].
      rewrite !andb_true_iff,
        (closures_entry_well_formed_iff index function_types closure_types
           value_slot_types).
      unfold closures_entry_shape_ok, entry_types_well_formed.
      rewrite !andb_true_iff. split.
      - intros [[Hslot (Hfit & (Hd1 & Hd2 & Hd3) & Hc & Hv)] Hext].
        split; [ | split; [ split | exact Hext ] ].
        + split; [ exact Hslot | ]. repeat split; try assumption.
          apply (proj2 (for_all_bindings_b_in Pos.eqb pos_eqb_decides_equality
                          _ closure_types Hd2)).
          intros g ty Hin.
          rewrite (proj1 (Hc g ty
                            (lookup_of_distinct_in Pos.eqb
                               pos_eqb_decides_equality closure_types Hd2 g
                               ty Hin))).
          reflexivity.
        + intros g ty Hin.
          exact (proj2 (Hc g ty
                          (lookup_of_distinct_in Pos.eqb
                             pos_eqb_decides_equality closure_types Hd2 g ty
                             Hin))).
        + intros w ty Hin.
          exact (Hv w ty
                   (lookup_of_distinct_in Pos.eqb pos_eqb_decides_equality
                      value_slot_types Hd3 w ty Hin)).
      - intros ([Hslot ((((Hfit & Hd1) & Hd2) & Hd3) & Hkinds)] & [Hc Hv]
                & Hext).
        split; [ split; [ exact Hslot | ] | exact Hext ].
        split; [ exact Hfit | ]. split; [ auto | ]. split.
        + intros g ty Hlookup.
          assert (Hin :=
            keyed_lookup_in Pos.eqb pos_eqb_decides_equality g ty
              closure_types Hlookup).
          split; [ | exact (Hc g ty Hin) ].
          apply kind_eqb_true.
          exact (proj1 (for_all_bindings_b_in Pos.eqb pos_eqb_decides_equality
                          _ closure_types Hd2)
                   Hkinds g ty Hin).
        + intros w ty Hlookup.
          exact (Hv w ty
                   (keyed_lookup_in Pos.eqb pos_eqb_decides_equality w ty
                      value_slot_types Hlookup)). }
    split.
    + intros [Hshape Hlists]. unfold closures_shape_ok in Hshape.
      apply andb_true_iff in Hshape as [Hdistinct Hshapes].
      assert (Hshapes' :=
        proj1 (for_all_bindings_b_in Pos.eqb pos_eqb_decides_equality _
                 known_closures Hdistinct)
          Hshapes).
      split.
      * intros [slot case] Hin.
        destruct (Hcases slot case Hin) as [Hentry Hext].
        apply all_at_app.
        destruct (proj1 (all_at_app _ _) (Hlists _ Hin)) as [He Hx].
        split;
          [ exact (proj1 (proj1 Hentry He)) | exact (proj1 (proj1 Hext Hx)) ].
      * apply (proj2 (row_like_for_closures_well_formed_iff known_closures)).
        split; [ exact Hdistinct | ]. intros slot case Hlookup.
        assert (Hin :=
          keyed_lookup_in Pos.eqb pos_eqb_decides_equality slot case
            known_closures Hlookup).
        destruct (Hcases slot case Hin) as [Hentry Hext].
        destruct (proj1 (all_at_app _ _) (Hlists _ Hin)) as [He Hx].
        apply (proj2 (Hcase_iff slot case)).
        split; [ exact (Hshapes' slot case Hin) | ].
        split;
          [ exact (proj2 (proj1 Hentry He)) | exact (proj2 (proj1 Hext Hx)) ].
    + intros [Hlists Hwf].
      destruct (proj1 (row_like_for_closures_well_formed_iff known_closures)
                  Hwf)
        as [Hdistinct Hwf_cases].
      assert (Hin_wf :
        ∀ (slot : function_slot) (case : row_like_closures_case),
          (slot, case) ∈ known_closures ->
          row_like_closures_case_well_formed slot case = true).
      { intros slot case Hin.
        exact (Hwf_cases slot case
                 (lookup_of_distinct_in Pos.eqb pos_eqb_decides_equality
                    known_closures Hdistinct slot case Hin)). }
      split.
      * unfold closures_shape_ok. apply andb_true_iff.
        split; [ exact Hdistinct | ].
        apply (proj2 (for_all_bindings_b_in Pos.eqb pos_eqb_decides_equality _
                        known_closures Hdistinct)).
        intros slot case Hin.
        exact (proj1 (proj1 (Hcase_iff slot case) (Hin_wf slot case Hin))).
      * intros [slot case] Hin.
        destruct (Hcases slot case Hin) as [Hentry Hext].
        destruct (proj1 (Hcase_iff slot case) (Hin_wf slot case Hin))
          as (_ & Htys & Hwf_ext).
        apply all_at_app.
        destruct (proj1 (all_at_app _ _) (Hlists _ Hin)) as [He Hx].
        split;
          [ exact (proj2 Hentry (conj He Htys))
          | exact (proj2 Hext (conj Hx Hwf_ext)) ].
  - (* An entry of closures. *)
    intros function_types closure_types value_slot_types Hclosures Hslots.
    simpl. rewrite !all_at_app, !all_at_flat_map. split.
    + intros [Hc Hv]. split; [ split | split ].
      * intros [g ty] Hin.
        exact (proj1 (proj1 (Hclosures g ty Hin) (Hc _ Hin))).
      * intros [w ty] Hin. exact (proj1 (proj1 (Hslots w ty Hin) (Hv _ Hin))).
      * intros g ty Hin.
        exact (proj2 (proj1 (Hclosures g ty Hin) (Hc _ Hin))).
      * intros w ty Hin. exact (proj2 (proj1 (Hslots w ty Hin) (Hv _ Hin))).
    + intros [[Hc Hv] [Hwc Hwv]]. split.
      * intros [g ty] Hin.
        exact (proj2 (Hclosures g ty Hin) (conj (Hc _ Hin) (Hwc g ty Hin))).
      * intros [w ty] Hin.
        exact (proj2 (Hslots w ty Hin) (conj (Hv _ Hin) (Hwv w ty Hin))).
  - (* An environment extension. *)
    intros equations Htypes. simpl.
    rewrite ?all_at_app, ?all_at_check, ?check_off, ?all_at_nil,
      ?all_at_flat_map.
    split.
    + intros [Hdistinct Heqs]. split.
      * intros [n ty] Hin. apply all_at_app.
        destruct (proj1 (all_at_app _ _) (Heqs _ Hin)) as [Hname Hty].
        split; [ exact Hname | exact (proj1 (proj1 (Htypes n ty Hin) Hty)) ].
      * apply (proj2 (env_extension_well_formed_iff equations)).
        split; [ exact Hdistinct | ]. intros n ty Hlookup.
        assert (Hin :=
          keyed_lookup_in name_eqb name_eqb_eq n ty equations Hlookup).
        destruct (proj1 (all_at_app _ _) (Heqs _ Hin)) as [_ Hty].
        exact (proj2 (proj1 (Htypes n ty Hin) Hty)).
    + intros [Heqs Hwf].
      destruct (proj1 (env_extension_well_formed_iff equations) Hwf)
        as [Hdistinct Hwf_tys].
      split; [ exact Hdistinct | ]. intros [n ty] Hin. apply all_at_app.
      destruct (proj1 (all_at_app _ _) (Heqs _ Hin)) as [Hname Hty].
      split; [ exact Hname | ]. apply (Htypes n ty Hin).
      split; [ exact Hty | ].
      exact (Hwf_tys n ty
               (lookup_of_distinct_in name_eqb name_eqb_eq equations Hdistinct
                  n ty Hin)).
Qed.

End Checked.

(* What is in a check: nothing, or [ill_formed] where the test fails. *)
Lemma in_check (with_checks test : bool) (p : simple * flambda_kind) :
  p ∈ check with_checks test ->
  with_checks = true ∧ test = false ∧ p = ill_formed.
Proof.
  unfold check. destruct with_checks, test; simpl; intros H;
    try contradiction.
  destruct H as [<- | []]. repeat split.
Qed.

(* A check that is kinded passes. *)
Lemma check_passes (P : simple -> flambda_kind -> Prop) (test : bool) :
  ¬ P (fst ill_formed) (snd ill_formed) ->
  all_at P (check true test) -> test = true.
Proof. intros Hill H. exact (proj1 (all_at_check P Hill test) H). Qed.

(* What a list with the equations or the checks asks, the aliases
   ask. *)
Lemma kinds_ask_aliases (P : simple -> flambda_kind -> Prop)
    (with_equations with_checks : bool) (t : flambda_type) :
  ¬ P (fst ill_formed) (snd ill_formed) ->
  all_at P (kinds_with with_equations with_checks t) ->
  all_at P (aliases_with_kinds t).
Proof.
  intros Hill H.
  assert (H' : all_at P (kinds_with with_equations false t)).
  { destruct with_checks; [ | exact H ].
    exact (proj1 (proj1 (proj1 (kinds_checked P Hill with_equations) t) H)). }
  destruct with_equations; [ | exact H' ].
  exact (proj1 (proj1 (kinds_with_equations_all P t) H')).
Qed.

Lemma kinds_of_head_ask_aliases (P : simple -> flambda_kind -> Prop)
    (with_equations with_checks : bool) (head : head_of_kind_value) :
  ¬ P (fst ill_formed) (snd ill_formed) ->
  all_at P (kinds_of_head_of_kind_value with_equations with_checks head) ->
  all_at P (aliases_of_head_of_kind_value head).
Proof.
  intros Hill H.
  assert (H' :
    all_at P (kinds_of_head_of_kind_value with_equations false head)).
  { destruct with_checks; [ | exact H ].
    exact (proj1 (proj1 (proj1 (proj2 (kinds_checked P Hill with_equations))
                           head)
                    H)). }
  destruct with_equations; [ | exact H' ].
  exact (proj1
           (proj1
              (kinds_all (kinds_of_head_of_kind_value true false)
                 aliases_of_head_of_kind_value
                 equation_names_of_head_of_kind_value P head
                 (fun p => proj1 (proj2 (kinds_with_equations p)) head))
              H')).
Qed.

Lemma kinds_with_unknown (with_equations with_checks : bool)
    (k : flambda_kind) :
  kinds_with with_equations with_checks (unknown k) = [].
Proof. destruct with_checks, k as [ | [] | | ]; reflexivity. Qed.

Lemma kinds_with_bottom (with_equations with_checks : bool)
    (k : flambda_kind) :
  kinds_with with_equations with_checks (bottom k) = [].
Proof. destruct with_checks, k as [ | [] | | ]; reflexivity. Qed.

Lemma kinds_with_alias (with_equations with_checks : bool) (k : flambda_kind)
    (s : simple) :
  kinds_with with_equations with_checks (alias_type_of k s) = [(s, k)].
Proof. destruct with_checks, k as [ | [] | | ]; reflexivity. Qed.

(* [ill_formed] is of no kind in an environment. *)
Lemma ill_formed_of_no_kind (tenv : Typing_env.t) :
  ¬ simple_of_kind tenv (fst ill_formed) (snd ill_formed).
Proof. intros H. cbn in H. discriminate H. Qed.

(* ======================================================================== *)
(* The interface: what a growing state keeps                                *)
(* ======================================================================== *)

Section Traversal.

Context (typing_env_well_formed : Typing_env.t -> Prop).

Context {state : Type}
  (joined_env : state -> Index.t -> option Typing_env.t)
  (definitions_of_existentials :
     state -> finite_map (Simples_in_joined_envs.t * flambda_kind))
  (kind_in_target_env : state -> variable -> option flambda_kind)
  (state_valid : state -> Prop)
  (state_kinded : state -> Prop).

Local Abbreviation envs_well_formed :=
  (joined_envs_well_formed typing_env_well_formed joined_env).
Local Abbreviation envs_kinded := (joined_envs_kinded joined_env).
Local Abbreviation grows :=
  (state_grows joined_env definitions_of_existentials kind_in_target_env).
Local Abbreviation extends :=
  (extends_input joined_env definitions_of_existentials).
Local Abbreviation args_in_scope := (join_args_in_scope joined_env).
Local Abbreviation exists_in_target :=
  (exists_in_target_env kind_in_target_env).
Local Abbreviation in_target := (answer_in_target kind_in_target_env).
Local Abbreviation kinded_in_target :=
  (answer_kinded_in_target joined_env kind_in_target_env).
Local Abbreviation of_kind_in_target :=
  (simple_of_kind_in_target joined_env kind_in_target_env).
Local Abbreviation step_sound :=
  (join_step_sound typing_env_well_formed joined_env
     definitions_of_existentials kind_in_target_env state_valid state_kinded).

Lemma grows_refl (s : state) : grows s s.
Proof. repeat split; auto. Qed.

Lemma grows_trans (s s' s'' : state) :
  grows s s' -> grows s' s'' -> grows s s''.
Proof.
  intros (Henv & Hdefs & Hkinds) (Henv' & Hdefs' & Hkinds').
  repeat split.
  - intros index. rewrite Henv'. apply Henv.
  - intros x definition Hx. apply Hdefs', Hdefs, Hx.
  - intros x kind Hx. apply Hkinds', Hkinds, Hx.
Qed.

Lemma grows_joined_env (s s' : state) (index : Index.t) :
  grows s s' -> joined_env s' index = joined_env s index.
Proof. intros (Henv & _). apply Henv. Qed.

Lemma grows_well_formed (s s' : state) :
  grows s s' -> envs_well_formed s -> envs_well_formed s'.
Proof.
  intros (Henv & _) Hwf index use Huse. rewrite Henv in Huse.
  exact (Hwf index use Huse).
Qed.

Lemma grows_kinded (s s' : state) :
  grows s s' -> envs_kinded s -> envs_kinded s'.
Proof.
  intros (Henv & _) Hkinded index use Huse. rewrite Henv in Huse.
  exact (Hkinded index use Huse).
Qed.

Lemma grows_args_in_scope {X : Type} (P : Typing_env.t -> X -> Prop)
    (s s' : state) (args : list (Index.t * X)) :
  grows s s' -> args_in_scope P s args -> args_in_scope P s' args.
Proof.
  intros (Henv & _) Hargs index x Hin.
  destruct (Hargs index x Hin) as (use & Huse & HP).
  exists use. rewrite Henv. split; assumption.
Qed.

(* A valuation that extends an input at a later state extends it at an
   earlier one. *)
Lemma grows_extends (s s' : state) (index : Index.t)
    (valuation valuation' : env) :
  grows s s' ->
  extends s' index valuation valuation' ->
  extends s index valuation valuation'.
Proof.
  intros (Henv & Hdefs & _) (use & Huse & Hagrees & Hholds).
  exists use. rewrite <- Henv. split; [ exact Huse | ].
  split; [ exact Hagrees | ].
  intros x simples kind simple v Hx.
  apply (Hholds x simples kind simple v). apply Hdefs. exact Hx.
Qed.

Lemma grows_exists (s s' : state) (x : variable) :
  grows s s' -> exists_in_target s x = true -> exists_in_target s' x = true.
Proof.
  intros (_ & _ & Hkinds). unfold exists_in_target_env.
  destruct (kind_in_target_env s x) as [kind | ] eqn:Hkind; [ | discriminate ].
  rewrite (Hkinds x kind Hkind). reflexivity.
Qed.

Lemma grows_in_target {Y : Type} (variables_of : Y -> list variable)
    (s s' : state) (y : Y) :
  grows s s' -> in_target variables_of s y -> in_target variables_of s' y.
Proof.
  intros Hgrows Htarget x Hx. exact (grows_exists _ _ _ Hgrows (Htarget x Hx)).
Qed.

Lemma grows_of_kind_in_target (s s' : state) (simple : simple)
    (kind : flambda_kind) :
  grows s s' ->
  of_kind_in_target s simple kind -> of_kind_in_target s' simple kind.
Proof.
  intros (Henv & _ & Hkinds). destruct simple as [x | sym | c]; simpl; auto.
  intros [Hk (index & use & Huse & Hmem)]. split; [ exact Hk | ].
  exists index, use. rewrite Henv. split; assumption.
Qed.

Lemma grows_kinded_in_target {Y : Type}
    (aliases_of : Y -> list (simple * flambda_kind)) (s s' : state) (y : Y) :
  grows s s' ->
  kinded_in_target aliases_of s y -> kinded_in_target aliases_of s' y.
Proof.
  intros Hgrows Hkinded simple kind Hin.
  exact (grows_of_kind_in_target _ _ _ _ Hgrows (Hkinded simple kind Hin)).
Qed.

(* ======================================================================== *)
(* Using a function that is sound in the middle of a run                    *)
(* ======================================================================== *)

Local Abbreviation arg_ok := arg_in_scope_and_kinded.

(* What a call that went from [s1] to [s2] gives at any later state:
   its answer is in the target, bounds each thing given, and is kinded
   if the state was. *)
Definition answer_holds_from {X Y : Type}
    (variables_of_answer : Y -> list variable)
    (aliases_of_answer : Y -> list (simple * flambda_kind))
    (bound : static_env -> env -> X -> env -> Y -> Prop) (s1 s2 : state)
    (args : list (Index.t * X)) (answer : Y) : Prop :=
  ∀ s' : state,
    grows s2 s' ->
    in_target variables_of_answer s' answer
    ∧ (∀ (index : Index.t) (x : X) (use : Typing_env.t)
         (statics : static_env) (valuation valuation' : env),
         In (index, x) args ->
         joined_env s' index = Some use ->
         typing_env_satisfied_strictly statics valuation use ->
         extends s' index valuation valuation' ->
         bound statics valuation x valuation' answer)
    ∧ (state_kinded s1 -> kinded_in_target aliases_of_answer s' answer).

(* A sound function called at a state [s1] that has grown from a state
   [s] at which its arguments were in scope. *)
Lemma step_sound_at {X Y : Type} (variables_of_arg : X -> list variable)
    (aliases_of_arg : X -> list (simple * flambda_kind))
    (variables_of_answer : Y -> list variable)
    (aliases_of_answer : Y -> list (simple * flambda_kind))
    (bound : static_env -> env -> X -> env -> Y -> Prop)
    (step : state -> list (Index.t * X) -> option (Y * state)) :
  step_sound variables_of_arg aliases_of_arg variables_of_answer
    aliases_of_answer bound step ->
  ∀ (s s1 s2 : state) (args : list (Index.t * X)) (answer : Y),
    envs_well_formed s -> envs_kinded s -> grows s s1 -> state_valid s1 ->
    join_args_distinct args ->
    args_in_scope (arg_ok variables_of_arg aliases_of_arg) s args ->
    step s1 args = Some (answer, s2) ->
    state_valid s2 ∧ grows s1 s2 ∧ (state_kinded s1 -> state_kinded s2)
    ∧ answer_holds_from variables_of_answer aliases_of_answer bound s1 s2 args
        answer.
Proof.
  intros Hsound s s1 s2 args answer Hwf Hkinded Hgrows Hvalid Hdistinct Hargs
    Hstep.
  destruct
    (Hsound s1 s2 args answer Hvalid (grows_well_formed _ _ Hgrows Hwf)
       (grows_kinded _ _ Hgrows Hkinded) Hdistinct
       (grows_args_in_scope _ _ _ _ Hgrows Hargs) Hstep)
    as (Hvalid2 & Hgrows2 & Htarget & Hbound & Hkinds).
  split; [ exact Hvalid2 | ]. split; [ exact Hgrows2 | ].
  split; [ intros Hk; apply Hkinds; exact Hk | ].
  intros s' Hgrows'.
  split; [ exact (grows_in_target _ _ _ _ Hgrows' Htarget) | ].
  split.
  - intros index x use statics valuation valuation' Hin Huse Hsatisfied
      Hextends.
    apply (Hbound index x use statics valuation valuation' Hin);
      [ | exact Hsatisfied | exact (grows_extends _ _ _ _ _ Hgrows' Hextends) ].
    rewrite <- (grows_joined_env _ _ index Hgrows'). exact Huse.
  - intros Hk.
    exact (grows_kinded_in_target _ _ _ _ Hgrows' (proj2 (Hkinds Hk))).
Qed.

(* Arguments made of parts of arguments that are in scope and kinded
   are in scope and kinded. *)
Lemma args_of_parts {X X' : Type} (variables_of : X -> list variable)
    (aliases_of : X -> list (simple * flambda_kind))
    (variables_of' : X' -> list variable)
    (aliases_of' : X' -> list (simple * flambda_kind)) (s : state)
    (args : list (Index.t * X)) (args' : list (Index.t * X')) :
  args_in_scope (arg_ok variables_of aliases_of) s args ->
  (∀ (index : Index.t) (x' : X'),
     In (index, x') args' ->
     ∃ x : X,
       In (index, x) args
       ∧ incl (variables_of' x') (variables_of x)
       ∧ incl (aliases_of' x') (aliases_of x)) ->
  args_in_scope (arg_ok variables_of' aliases_of') s args'.
Proof.
  intros Hargs Hparts index x' Hin.
  destruct (Hparts index x' Hin) as (x & Hin' & Hvariables & Haliases).
  destruct (Hargs index x Hin') as (use & Huse & Hknown & Hkinds).
  exists use. split; [ exact Huse | ].
  split.
  - intros y Hy. apply Hknown, Hvariables, Hy.
  - intros simple kind Hsimple. apply Hkinds, Haliases, Hsimple.
Qed.

(* A sound function is sound for a weaker bound. *)
Lemma step_sound_bound {X Y : Type} (variables_of_arg : X -> list variable)
    (aliases_of_arg : X -> list (simple * flambda_kind))
    (variables_of_answer : Y -> list variable)
    (aliases_of_answer : Y -> list (simple * flambda_kind))
    (bound bound' : static_env -> env -> X -> env -> Y -> Prop)
    (step : state -> list (Index.t * X) -> option (Y * state)) :
  (∀ (statics : static_env) (valuation : env) (x : X) (valuation' : env)
     (y : Y),
     bound statics valuation x valuation' y ->
     bound' statics valuation x valuation' y) ->
  step_sound variables_of_arg aliases_of_arg variables_of_answer
    aliases_of_answer bound step ->
  step_sound variables_of_arg aliases_of_arg variables_of_answer
    aliases_of_answer bound' step.
Proof.
  intros Himpl Hsound s s' args answer Hvalid Hwf Hkinded Hdistinct Hargs
    Hstep.
  destruct (Hsound s s' args answer Hvalid Hwf Hkinded Hdistinct Hargs Hstep)
    as (Hvalid' & Hgrows & Htarget & Hbound & Hkinds).
  split; [ exact Hvalid' | ]. split; [ exact Hgrows | ].
  split; [ exact Htarget | ]. split; [ | exact Hkinds ].
  intros index x use statics valuation valuation' Hin Huse Hsatisfied
    Hextends.
  apply Himpl.
  exact
    (Hbound index x use statics valuation valuation' Hin Huse Hsatisfied
       Hextends).
Qed.

(* An answer that holds from a state holds from a later one. *)
Lemma answer_holds_from_later {X Y : Type}
    (variables_of_answer : Y -> list variable)
    (aliases_of_answer : Y -> list (simple * flambda_kind))
    (bound : static_env -> env -> X -> env -> Y -> Prop)
    (s0 s1 s2 s3 : state) (args : list (Index.t * X)) (answer : Y) :
  (state_kinded s0 -> state_kinded s1) ->
  grows s2 s3 ->
  answer_holds_from variables_of_answer aliases_of_answer bound s1 s2 args
    answer ->
  answer_holds_from variables_of_answer aliases_of_answer bound s0 s3 args
    answer.
Proof.
  intros Hk Hgrows Hholds s' Hgrows'.
  destruct (Hholds s' (grows_trans _ _ _ Hgrows Hgrows'))
    as (Htarget & Hbound & Hkinds).
  split; [ exact Htarget | ]. split; [ exact Hbound | ].
  intros Hk0. exact (Hkinds (Hk Hk0)).
Qed.

(* A join that may answer bottom: it answers bottom only of no thing,
   and otherwise as a sound function does, called at a state [s1] that
   has grown from a state [s] at which the things given were in scope
   and kinded. *)
Definition case_join_sound {X Y : Type} (variables_of_arg : X -> list variable)
    (aliases_of_arg : X -> list (simple * flambda_kind))
    (variables_of_answer : Y -> list variable)
    (aliases_of_answer : Y -> list (simple * flambda_kind))
    (bound : static_env -> env -> X -> env -> Y -> Prop)
    (join : state -> list (Index.t * X) -> option (Or_bottom.t (Y * state)))
    : Prop :=
  ∀ (s s1 : state) (args : list (Index.t * X))
    (result : Or_bottom.t (Y * state)),
    envs_well_formed s -> envs_kinded s -> grows s s1 -> state_valid s1 ->
    join_args_distinct args ->
    args_in_scope (arg_ok variables_of_arg aliases_of_arg) s args ->
    join s1 args = Some result ->
    match result with
    | Or_bottom.Bottom => args = []
    | Or_bottom.Ok (answer, s2) =>
      state_valid s2 ∧ grows s1 s2 ∧ (state_kinded s1 -> state_kinded s2)
      ∧ answer_holds_from variables_of_answer aliases_of_answer bound s1 s2
          args answer
    end.

(* ======================================================================== *)
(* [generic_merge_map_known]                                                *)
(* ======================================================================== *)

Section Merge_map_known.

Context {K value other : Type} (eqb : K -> K -> bool)
  (Heqb : decides_equality eqb) (of_other : other -> value)
  (variables_of : value -> list variable)
  (aliases_of : value -> list (simple * flambda_kind))
  (bound : static_env -> env -> value -> env -> value -> Prop)
  (join :
     state -> list (Index.t * value) -> option (Or_bottom.t (value * state))).

Hypothesis Hjoin :
  case_join_sound variables_of aliases_of variables_of aliases_of bound join.

(* The cases of a row-like structure are in scope and kinded in an
   environment. *)
Definition known_in_scope (use : Typing_env.t)
    (m : Keyed.finite_map K value * Or_bottom.t other) : Prop :=
  (∀ (key : K) (v : value),
     Keyed.lookup eqb key (fst m) = Some v ->
     arg_ok variables_of aliases_of use v)
  ∧ ∀ o : other,
      snd m = Or_bottom.Ok o -> arg_ok variables_of aliases_of use (of_other o).

(* The map answered has no key twice; what it binds a key to is the
   join of the cases gathered for the key; and it binds every key for
   which cases were gathered. *)
Lemma generic_merge_map_known_is_sound (s s1 s2 : state)
    (maps : list (Index.t * (Keyed.finite_map K value * Or_bottom.t other)))
    (joined : Keyed.finite_map K value) :
  envs_well_formed s -> envs_kinded s -> grows s s1 -> state_valid s1 ->
  join_args_distinct maps ->
  args_in_scope known_in_scope s maps ->
  generic_merge_map_known eqb of_other join s1 maps = Some (joined, s2) ->
  state_valid s2 ∧ grows s1 s2 ∧ (state_kinded s1 -> state_kinded s2)
  ∧ (∀ (key : K) (y : value),
       In (key, y) joined -> Keyed.lookup eqb key joined = Some y)
  ∧ (∀ (key : K) (y : value),
       Keyed.lookup eqb key joined = Some y ->
       ∃ vals : list (Index.t * value),
         Keyed.lookup eqb key (gather_known eqb of_other maps) = Some vals
         ∧ answer_holds_from variables_of aliases_of bound s1 s2 vals y)
  ∧ ∀ key : K,
      Keyed.lookup eqb key joined = None ->
      Keyed.lookup eqb key (gather_known eqb of_other maps) = None.
Proof.
  intros Hwf Hkinded Hgrows Hvalid Hdistinct Hargs Hmerge.
  unfold generic_merge_map_known in Hmerge.
  destruct maps as [ | first rest].
  { injection Hmerge as <- <-. split; [ exact Hvalid | ].
    split; [ apply grows_refl | ]. split; [ auto | ].
    split; [ intros ? ? [] | ]. split; [ intros ? ? [=] | reflexivity ]. }
  cbv beta iota in Hmerge.
  set (P := fun s' : state => state_valid s' ∧ grows s s').
  set (R :=
    fun s' s'' : state =>
      grows s' s'' ∧ (state_kinded s' -> state_kinded s'')).
  assert (Hrefl : ∀ s' : state, R s' s').
  { intros s'. split; [ apply grows_refl | auto ]. }
  assert (Htrans : ∀ a b c : state, R a b -> R b c -> R a c).
  { intros a b c [Hab Hkab] [Hbc Hkbc].
    split; [ exact (grows_trans _ _ _ Hab Hbc) | auto ]. }
  assert (Hvals :
    ∀ (key : K) (vals : list (Index.t * value)),
      Keyed.lookup eqb key (gather_known eqb of_other (first :: rest))
      = Some vals ->
      join_args_distinct vals
      ∧ args_in_scope (arg_ok variables_of aliases_of) s vals
      ∧ vals <> []).
  { intros key vals Hkey.
    destruct
      (gather_known_spec eqb Heqb of_other (first :: rest) key vals Hdistinct
         Hkey)
      as (Hd & Hfrom & _ & Hne).
    split; [ exact Hd | ]. split; [ | exact Hne ].
    intros index v Hin. destruct (Hfrom index v Hin) as (i & Hi & Hfst & Hv).
    destruct i as [index' m]. simpl in Hfst. subst index'.
    destruct (Hargs index m Hi) as (use & Huse & Hknown & Hother).
    exists use. split; [ exact Huse | ].
    destruct Hv as [Hv | (_ & o & Ho & ->)];
      [ exact (Hknown key v Hv) | exact (Hother o Ho) ]. }
  eapply (filter_map_fold_spec eqb Heqb P R Hrefl Htrans) in Hmerge.
  - destruct Hmerge
      as ((Hvalid2 & Hgrows02) & (Hgrows2 & Hk2) & Hin & Hfound & Hall).
    split; [ exact Hvalid2 | ]. split; [ exact Hgrows2 | ].
    split; [ exact Hk2 | ]. split; [ exact Hin | ].
    split.
    + intros key y Hy.
      destruct (Hfound key y Hy)
        as (vals & s1' & s2' & Hvals' & (Hvalid1' & Hgrows1')
            & (Hgrows11 & Hk11) & Hg & (Hgrows22 & _)).
      exists vals. split; [ exact Hvals' | ].
      destruct (Hvals key vals Hvals') as (Hd & Hscope & _).
      destruct (join s1' vals) as [[[y' s2''] | ] | ] eqn:Hj;
        try discriminate.
      injection Hg as -> ->.
      pose proof
        (Hjoin s s1' vals _ Hwf Hkinded Hgrows1' Hvalid1' Hd Hscope Hj)
        as (_ & _ & _ & Hholds).
      exact
        (answer_holds_from_later _ _ _ s1 s1' s2' s2 vals y Hk11 Hgrows22
           Hholds).
    + intros key Hnone.
      destruct
        (Keyed.lookup eqb key (gather_known eqb of_other (first :: rest)))
        as [vals | ] eqn:Hvals'; [ | reflexivity ].
      exfalso.
      destruct (Hall key vals Hvals')
        as [[y Hy] | (s1' & s2' & (Hvalid1' & Hgrows1') & _ & Hg)];
        [ congruence | ].
      destruct (Hvals key vals Hvals') as (Hd & Hscope & Hne).
      destruct (join s1' vals) as [[[y' s2''] | ] | ] eqn:Hj;
        try discriminate.
      exact
        (Hne (Hjoin s s1' vals _ Hwf Hkinded Hgrows1' Hvalid1' Hd Hscope Hj)).
  - intros key vals s1' a s2' Hvals' (Hvalid1' & Hgrows1') Hg.
    destruct (Hvals key vals Hvals') as (Hd & Hscope & _).
    destruct (join s1' vals) as [[[y' s2''] | ] | ] eqn:Hj; try discriminate.
    + injection Hg as <- <-.
      pose proof
        (Hjoin s s1' vals _ Hwf Hkinded Hgrows1' Hvalid1' Hd Hscope Hj)
        as (Hvalid2' & Hgrows2' & Hk2' & _).
      split;
        [ split;
          [ exact Hvalid2' | exact (grows_trans _ _ _ Hgrows1' Hgrows2') ]
        | split; assumption ].
    + injection Hg as <- <-. split; [ split; assumption | apply Hrefl ].
  - split; [ exact Hvalid | exact Hgrows ].
Qed.

End Merge_map_known.

(* ======================================================================== *)
(* The keys and the cases of the map that a merge answers                   *)
(* ======================================================================== *)

(* The keys that a fold answers are among those that it went through,
   and it answers none twice. *)
Lemma tabulate_fold_opt_keys {K A S : Type} (eqb : K -> K -> bool)
    (f : K -> S -> option (option A * S)) (keys : list K) :
  ∀ (s : S) (out : Keyed.finite_map K A) (s' : S),
    Keyed.distinct eqb keys = true ->
    Keyed.tabulate_fold_opt f keys s = Some (out, s') ->
    Keyed.distinct eqb (map fst out) = true
    ∧ ∀ key : K,
        Keyed.among eqb key (map fst out) = true ->
        Keyed.among eqb key keys = true.
Proof.
  induction keys as [ | key keys IH]; intros s out s' Hdistinct Hfold;
    simpl in Hfold.
  - injection Hfold as <- <-. split; [ reflexivity | intros key H; exact H ].
  - destruct (f key s) as [[a s1] | ]; [ | discriminate ].
    destruct (Keyed.tabulate_fold_opt f keys s1) as [[bindings s2] | ]
      eqn:Hrest; [ | discriminate ].
    injection Hfold as <- <-. simpl in Hdistinct.
    apply andb_true_iff in Hdistinct as [Hfresh Hdistinct].
    destruct (IH s1 bindings s2 Hdistinct Hrest) as [Hout Hsub].
    destruct a as [a | ].
    + split.
      * simpl. rewrite Hout, andb_true_r. apply negb_true_iff.
        destruct (Keyed.among eqb key (map fst bindings)) eqn:Hamong;
          [ | reflexivity ].
        rewrite (Hsub key Hamong) in Hfresh. discriminate Hfresh.
      * intros key' Hamong. simpl in Hamong |- *.
        apply orb_true_iff in Hamong as [H | H];
          [ rewrite H; reflexivity | ].
        rewrite (Hsub key' H). apply orb_true_r.
    + split; [ exact Hout | ]. intros key' Hamong. simpl.
      rewrite (Hsub key' Hamong). apply orb_true_r.
Qed.

Section Merge_shape.

Context {K value other : Type} (eqb : K -> K -> bool)
  (Heqb : decides_equality eqb) (of_other : other -> value)
  (join :
     state -> list (Index.t * value) -> option (Or_bottom.t (value * state)))
  (Q : K -> value -> Prop).

(* The map answered has no key twice, and what it binds a key to has a
   property that the cases gathered for the key have and that the join
   keeps. *)
Lemma generic_merge_map_known_shape (s s' : state)
    (maps : list (Index.t * (Keyed.finite_map K value * Or_bottom.t other)))
    (joined : Keyed.finite_map K value) :
  join_args_distinct maps ->
  (∀ (key : K) (s1 s2 : state) (vals : list (Index.t * value)) (v : value),
     join s1 vals = Some (Or_bottom.Ok (v, s2)) ->
     (∀ (index : Index.t) (v0 : value), In (index, v0) vals -> Q key v0) ->
     Q key v) ->
  (∀ (index : Index.t) (m : Keyed.finite_map K value * Or_bottom.t other),
     In (index, m) maps ->
     (∀ (key : K) (v : value),
        Keyed.lookup eqb key (fst m) = Some v -> Q key v)
     ∧ ∀ (o : other) (key : K),
         snd m = Or_bottom.Ok o -> Q key (of_other o)) ->
  generic_merge_map_known eqb of_other join s maps = Some (joined, s') ->
  Keyed.distinct eqb (map fst joined) = true
  ∧ ∀ (key : K) (v : value),
      Keyed.lookup eqb key joined = Some v -> Q key v.
Proof.
  intros Hdistinct Hjoin Hmaps Hmerge. unfold generic_merge_map_known in Hmerge.
  destruct maps as [ | m0 maps0].
  { injection Hmerge as <- <-.
    split; [ reflexivity | intros key v H; discriminate H ]. }
  set (maps := m0 :: maps0) in *.
  match type of Hmerge with
  | Keyed.filter_map_fold _ ?g0 _ _ = _ => set (g := g0) in Hmerge
  end.
  split.
  - unfold Keyed.filter_map_fold, Keyed.filter_map_fold_on in Hmerge.
    exact (proj1 (tabulate_fold_opt_keys eqb _ _ s joined s'
                    (keyed_domain_distinct K eqb Heqb _ _) Hmerge)).
  - intros key v Hv.
    destruct
      (filter_map_fold_spec eqb Heqb (fun _ : state => True)
         (fun _ _ : state => True) (fun _ => I) (fun _ _ _ _ _ => I) g
         (gather_known eqb of_other maps)
         (fun _ _ _ _ _ _ _ _ => conj I I) s joined s' I Hmerge)
      as (_ & _ & _ & Hfound & _).
    destruct (Hfound key v Hv) as (vals & s1 & s2 & Hvals & _ & _ & Hg & _).
    unfold g in Hg.
    destruct (join s1 vals) as [[[v' s3] | ] | ] eqn:Hj; try discriminate.
    injection Hg as <- <-. apply (Hjoin key s1 s3 vals v' Hj).
    intros index v0 Hin.
    destruct (gather_known_spec eqb Heqb of_other maps key vals Hdistinct Hvals)
      as (_ & Hfrom & _).
    destruct (Hfrom index v0 Hin)
      as ([index' m] & Hi & _ & [Hl | (_ & o & Ho & ->)]).
    + exact (proj1 (Hmaps index' m Hi) key v0 Hl).
    + exact (proj2 (Hmaps index' m Hi) o key Ho).
Qed.

End Merge_shape.

(* The contract of the join of types is that of a sound function. *)
Definition type_bound :=
  known_bound (all_kept type_satisfied type_satisfied).

(* ======================================================================== *)
(* [n_way_join_unknown]                                                     *)
(* ======================================================================== *)

Theorem n_way_join_unknown_is_sound :
  n_way_join_unknown_sound typing_env_well_formed joined_env
    definitions_of_existentials kind_in_target_env state_valid state_kinded.
Proof.
  intros A B variables_of_arg aliases_of_arg variables_of_answer
    aliases_of_answer bound n_way_join_contents Hcontents s s' args answer
    Hvalid Hwf Hkinded Hdistinct Hargs Hstep.
  unfold n_way_join_unknown in Hstep.
  destruct (known_contents [] args) as [contents | ] eqn:Hknown.
  - destruct (known_contents_spec args [] contents Hknown)
      as (Hfst & Hfrom & Hto & _).
    assert (Hdistinct' : join_args_distinct contents).
    { unfold join_args_distinct. rewrite Hfst, app_nil_r.
      apply NoDup_rev. exact Hdistinct. }
    assert (Hargs' :
      args_in_scope (arg_in_scope_and_kinded variables_of_arg aliases_of_arg)
        s contents).
    { intros index a Hin. destruct (Hfrom index a Hin) as [[] | Hin'].
      exact (Hargs index (Or_unknown.Known a) Hin'). }
    destruct
      (Hcontents s s' contents answer Hvalid Hwf Hkinded Hdistinct' Hargs'
         Hstep)
      as (Hvalid' & Hgrows & Htarget & Hbound & Hkinds).
    split; [ exact Hvalid' | ]. split; [ exact Hgrows | ].
    split; [ exact Htarget | ]. split; [ | exact Hkinds ].
    intros index x use statics valuation valuation' Hin Huse Hsatisfied
      Hextends.
    destruct (Hto index x Hin) as (a & -> & Hin').
    specialize
      (Hbound index a use statics valuation valuation' Hin' Huse Hsatisfied
         Hextends).
    destruct answer; exact Hbound.
  - injection Hstep as <- <-.
    split; [ exact Hvalid | ]. split; [ apply grows_refl | ].
    split; [ intros x [] | ].
    split.
    + intros index x use statics valuation valuation' _ _ _ _.
      destruct x; exact I.
    + intros Hk. split; [ exact Hk | intros simple kind [] ].
Qed.

(* ======================================================================== *)
(* The functions that call the join of [Simple]s                            *)
(* ======================================================================== *)

Section Simples.

Context
  (n_way_join_simples :
     state -> flambda_kind -> list (Index.t * simple)
     -> option (Or_bottom.t simple * state)).

Hypothesis Hsimples :
  n_way_join_simples_sound typing_env_well_formed joined_env
    definitions_of_existentials kind_in_target_env state_valid state_kinded
    n_way_join_simples.

Theorem n_way_join_relation_simples_is_sound :
  n_way_join_relation_simples_sound typing_env_well_formed joined_env
    definitions_of_existentials kind_in_target_env state_valid state_kinded
    n_way_join_simples.
Proof.
  intros s s' simples x Hvalid Hwf Hkinded Hdistinct Hargs Hstep.
  unfold n_way_join_relation_simples in Hstep.
  destruct (n_way_join_simples s (Naked_number Kind_naked_immediate) simples)
    as [[joined s1] | ] eqn:Hjoin; [ | discriminate ].
  assert (Hkinds :
    args_in_scope
      (fun use simple =>
         simple_of_kind use simple (Naked_number Kind_naked_immediate))
      s simples).
  { intros index simple Hin.
    destruct (Hargs index simple Hin) as (use & Huse & _ & Haliases).
    exists use. split; [ exact Huse | ]. apply Haliases. left. reflexivity. }
  destruct
    (Hsimples s s1 _ simples joined Hvalid Hwf Hkinded Hdistinct Hkinds Hjoin)
    as (Hvalid1 & Hgrows & _ & _ & Hok & Hkinded1).
  assert (Hnone :
    Some (None, s1) = Some (x, s') ->
    state_valid s' ∧ grows s s'
    ∧ in_target free_variables_of_option s' x
    ∧ (∀ (index : Index.t) (simple : FlambdaModel.simple)
         (use : Typing_env.t) (statics : static_env)
         (valuation valuation' : env),
         In (index, simple) simples ->
         joined_env s' index = Some use ->
         typing_env_satisfied_strictly statics valuation use ->
         extends s' index valuation valuation' ->
         ∀ n : Z,
           eval_simple valuation simple
           = Some (Value_const (Naked_immediate n)) ->
           variable_is valuation' x n)
    ∧ (state_kinded s ->
       state_kinded s' ∧ kinded_in_target aliases_of_relation s' x)).
  { intros [= <- <-].
    split; [ exact Hvalid1 | ]. split; [ exact Hgrows | ].
    split; [ intros y [] | ]. split; [ intros; exact I | ].
    intros Hk. split; [ apply Hkinded1; exact Hk | intros simple kind [] ]. }
  destruct joined as [[var | sym | c] | ];
    [ | apply Hnone; exact Hstep .. ].
  injection Hstep as <- <-.
  destruct (Hok (Var var) eq_refl) as (Htarget & Hbound).
  split; [ exact Hvalid1 | ]. split; [ exact Hgrows | ].
  split; [ exact Htarget | ].
  split.
  - intros index simple use statics valuation valuation' Hin Huse Hsatisfied
      Hextends n Heval.
    exact
      (Hbound index simple use statics valuation valuation' _ Hin Huse
         Hsatisfied Hextends Heval).
  - intros Hk. split; [ apply Hkinded1; exact Hk | ].
    intros simple kind [[= <- <-] | []].
    apply Hkinded1; [ exact Hk | reflexivity ].
Qed.

(* What the [Simple] of a head says of null, for [is_null_simples]. *)
Definition is_null_simple_of (head : head_of_kind_value) (simple : simple)
    : Prop :=
  match is_null_of_head head with
  | Not_null => simple = untagged_const_bool false
  | Maybe_null (Some x) => simple = Var x
  | Maybe_null None => False
  end.

Lemma is_null_simples_spec (heads : list (Index.t * head_of_kind_value)) :
  ∀ simples : list (Index.t * simple),
    is_null_simples heads = Some simples ->
    map fst simples = map fst heads
    ∧ (∀ (index : Index.t) (head : head_of_kind_value),
         In (index, head) heads ->
         ∃ simple : simple,
           In (index, simple) simples ∧ is_null_simple_of head simple)
    ∧ ∀ (index : Index.t) (simple : simple),
        In (index, simple) simples ->
        ∃ head : head_of_kind_value,
          In (index, head) heads ∧ is_null_simple_of head simple.
Proof.
  induction heads as [ | [id [non_null is_null]] heads IH];
    intros simples Hsimples'; simpl in Hsimples'.
  - injection Hsimples' as <-. repeat split; [ intros ? ? [] | intros ? ? [] ].
  - assert (Hcons :
      ∀ simple : simple,
        is_null_simple_of (Head_of_kind_value non_null is_null) simple ->
        option_map (cons (id, simple)) (is_null_simples heads) = Some simples ->
        map fst simples = map fst ((id, Head_of_kind_value non_null is_null)
                                   :: heads)
        ∧ (∀ (index : Index.t) (head : head_of_kind_value),
             In (index, head)
               ((id, Head_of_kind_value non_null is_null) :: heads) ->
             ∃ simple : FlambdaModel.simple,
               In (index, simple) simples ∧ is_null_simple_of head simple)
        ∧ ∀ (index : Index.t) (simple : FlambdaModel.simple),
            In (index, simple) simples ->
            ∃ head : head_of_kind_value,
              In (index, head)
                ((id, Head_of_kind_value non_null is_null) :: heads)
              ∧ is_null_simple_of head simple).
    { intros simple Hsimple Hrest.
      destruct (is_null_simples heads) as [rest | ]; [ | discriminate ].
      injection Hrest as <-.
      destruct (IH rest eq_refl) as (Hfst & Hto & Hfrom).
      repeat split.
      - simpl. rewrite Hfst. reflexivity.
      - intros index head [[= <- <-] | Hin].
        + exists simple. split; [ left; reflexivity | exact Hsimple ].
        + destruct (Hto index head Hin) as (simple' & Hin' & Hsimple').
          exists simple'. split; [ right; exact Hin' | exact Hsimple' ].
      - intros index simple' [[= <- <-] | Hin].
        + eexists. split; [ left; reflexivity | exact Hsimple ].
        + destruct (Hfrom index simple' Hin) as (head & Hin' & Hsimple').
          exists head. split; [ right; exact Hin' | exact Hsimple' ]. }
    destruct is_null as [ | [x | ]].
    + exact (Hcons _ eq_refl Hsimples').
    + exact (Hcons _ eq_refl Hsimples').
    + discriminate.
Qed.

Theorem n_way_join_is_null_is_sound :
  n_way_join_is_null_sound typing_env_well_formed joined_env
    definitions_of_existentials kind_in_target_env state_valid state_kinded
    n_way_join_simples.
Proof.
  intros s s' heads is_null Hvalid Hwf Hkinded Hdistinct Hargs Hstep.
  unfold n_way_join_is_null in Hstep.
  (* The answer that says nothing. *)
  assert (Hnone :
    ∀ s1 : state,
      state_valid s1 -> grows s s1 -> (state_kinded s -> state_kinded s1) ->
      Some (Maybe_null None, s1) = Some (is_null, s') ->
      state_valid s' ∧ grows s s'
      ∧ in_target variables_of_is_null s' is_null
      ∧ (∀ (index : Index.t) (head : head_of_kind_value)
           (use : Typing_env.t) (statics : static_env)
           (valuation valuation' : env),
           In (index, head) heads ->
           joined_env s' index = Some use ->
           typing_env_satisfied_strictly statics valuation use ->
           extends s' index valuation valuation' ->
           ∀ null : bool,
             is_null_says valuation (is_null_of_head head) null ->
             is_null_says valuation' is_null null)
      ∧ (state_kinded s ->
         state_kinded s' ∧ kinded_in_target aliases_of_is_null s' is_null)).
  { intros s1 Hvalid1 Hgrows Hk [= <- <-].
    split; [ exact Hvalid1 | ]. split; [ exact Hgrows | ].
    split; [ intros y [] | ]. split; [ intros; exact I | ].
    intros Hk'. split; [ exact (Hk Hk') | intros simple kind [] ]. }
  destruct (is_null_simples heads) as [simples | ] eqn:Hsimples';
    [ | exact (Hnone s Hvalid (grows_refl s) (fun H => H) Hstep) ].
  destruct (is_null_simples_spec heads simples Hsimples')
    as (Hfst & Hto & Hfrom).
  destruct (n_way_join_simples s (Naked_number Kind_naked_immediate) simples)
    as [[joined s1] | ] eqn:Hjoin; [ | discriminate ].
  assert (Hdistinct' : join_args_distinct simples).
  { unfold join_args_distinct. rewrite Hfst. exact Hdistinct. }
  assert (Hkinds :
    args_in_scope
      (fun use simple =>
         simple_of_kind use simple (Naked_number Kind_naked_immediate))
      s simples).
  { intros index simple Hin.
    destruct (Hfrom index simple Hin) as ([non_null is_null'] & Hin' & Hsimple).
    destruct (Hargs index _ Hin') as (use & Huse & _ & Haliases).
    exists use. split; [ exact Huse | ].
    unfold is_null_simple_of in Hsimple. simpl in Hsimple.
    destruct is_null' as [ | [x | ]]; [ | | contradiction ]; subst simple.
    - reflexivity.
    - apply Haliases. simpl. apply in_or_app. right. left. reflexivity. }
  destruct
    (Hsimples s s1 _ simples joined Hvalid Hwf Hkinded Hdistinct' Hkinds Hjoin)
    as (Hvalid1 & Hgrows & _ & _ & Hok & Hkinded1).
  assert (Hk1 : state_kinded s -> state_kinded s1).
  { intros Hk. apply Hkinded1. exact Hk. }
  (* What the join of the [Simple]s gives for a head. *)
  assert (Hjoined :
    ∀ (joined' : simple) (index : Index.t) (head : head_of_kind_value)
      (use : Typing_env.t) (statics : static_env) (valuation valuation' : env)
      (null : bool),
      joined = Or_bottom.Ok joined' ->
      In (index, head) heads ->
      joined_env s1 index = Some use ->
      typing_env_satisfied_strictly statics valuation use ->
      extends s1 index valuation valuation' ->
      is_null_says valuation (is_null_of_head head) null ->
      eval_simple valuation' joined'
      = Some (Value_const (Naked_immediate (if null then 1%Z else 0%Z)))).
  { intros joined' index head use statics valuation valuation' null Hjoined'
      Hin Huse Hsatisfied Hextends Hsays.
    destruct (Hok joined' Hjoined') as (_ & Hbound).
    destruct (Hto index head Hin) as (simple & Hin' & Hsimple).
    apply
      (Hbound index simple use statics valuation valuation' _ Hin' Huse
         Hsatisfied Hextends).
    unfold is_null_simple_of in Hsimple.
    destruct (is_null_of_head head) as [ | [x | ]];
      [ | | contradiction ]; subst simple.
    - simpl in Hsays. subst null. reflexivity.
    - exact Hsays. }
  destruct joined as [[var | sym | c] | ].
  - (* A variable. *)
    injection Hstep as <- <-.
    destruct (Hok (Var var) eq_refl) as (Htarget & _).
    split; [ exact Hvalid1 | ]. split; [ exact Hgrows | ].
    split; [ exact Htarget | ].
    split.
    + intros index head use statics valuation valuation' Hin Huse Hsatisfied
        Hextends null Hsays.
      exact
        (Hjoined _ index head use statics valuation valuation' null eq_refl
           Hin Huse Hsatisfied Hextends Hsays).
    + intros Hk. split; [ exact (Hk1 Hk) | ].
      intros simple kind [[= <- <-] | []].
      apply Hkinded1; [ exact Hk | reflexivity ].
  - discriminate.
  - (* A constant. *)
    assert (Hfalse :
      ∀ i : Z,
        c = Naked_immediate i -> Z.eqb i 0 = true ->
        Some (Not_null, s1) = Some (is_null, s') ->
        state_valid s' ∧ grows s s'
        ∧ in_target variables_of_is_null s' is_null
        ∧ (∀ (index : Index.t) (head : head_of_kind_value)
             (use : Typing_env.t) (statics : static_env)
             (valuation valuation' : env),
             In (index, head) heads ->
             joined_env s' index = Some use ->
             typing_env_satisfied_strictly statics valuation use ->
             extends s' index valuation valuation' ->
             ∀ null : bool,
               is_null_says valuation (is_null_of_head head) null ->
               is_null_says valuation' is_null null)
        ∧ (state_kinded s ->
           state_kinded s' ∧ kinded_in_target aliases_of_is_null s' is_null)).
    { intros i -> Hzero [= <- <-]. apply Z.eqb_eq in Hzero. subst i.
      split; [ exact Hvalid1 | ]. split; [ exact Hgrows | ].
      split; [ intros y [] | ].
      split.
      - intros index head use statics valuation valuation' Hin Huse
          Hsatisfied Hextends null Hsays.
        pose proof
          (Hjoined _ index head use statics valuation valuation' null eq_refl
             Hin Huse Hsatisfied Hextends Hsays) as Heval.
        simpl in Heval. destruct null; [ discriminate | reflexivity ].
      - intros Hk. split; [ exact (Hk1 Hk) | intros simple kind [] ]. }
    destruct c as [i | | | | | | | | | | | | | | ];
      try exact (Hnone s1 Hvalid1 Hgrows Hk1 Hstep).
    destruct (Z.eqb i 0) eqn:Hzero.
    + exact (Hfalse i eq_refl Hzero Hstep).
    + exact (Hnone s1 Hvalid1 Hgrows Hk1 Hstep).
  - exact (Hnone s1 Hvalid1 Hgrows Hk1 Hstep).
Qed.

End Simples.

(* ======================================================================== *)
(* Numbers                                                                  *)
(* ======================================================================== *)

Lemma fold_union_keeps (ts : list (Index.t * list Z)) :
  ∀ (acc : list Z) (n : Z),
    n ∈ acc ->
    n ∈ List.fold_left (fun t1 t2 => numbers_union t1 (snd t2)) ts acc.
Proof.
  induction ts as [ | [id t] ts IH]; intros acc n Hin; simpl; [ exact Hin | ].
  apply IH. apply numbers_union_in. left. exact Hin.
Qed.

Lemma fold_union_in (ts : list (Index.t * list Z)) :
  ∀ (acc : list Z) (index : Index.t) (t : list Z) (n : Z),
    In (index, t) ts -> n ∈ t ->
    n ∈ List.fold_left (fun t1 t2 => numbers_union t1 (snd t2)) ts acc.
Proof.
  induction ts as [ | [id t'] ts IH]; intros acc index t n Hin Hn; simpl;
    [ destruct Hin | ].
  destruct Hin as [[= <- <-] | Hin].
  - apply fold_union_keeps. apply numbers_union_in. right. exact Hn.
  - exact (IH _ index t n Hin Hn).
Qed.

Theorem n_way_join_head_of_kind_naked_numbers_is_sound :
  n_way_join_head_of_kind_naked_numbers_sound typing_env_well_formed
    joined_env definitions_of_existentials kind_in_target_env state_valid
    state_kinded.
Proof.
  intros k s s' args answer Hvalid Hwf Hkinded Hdistinct Hargs Hstep.
  destruct args as [ | [id1 head1] heads]; [ discriminate | ].
  injection Hstep as <- <-.
  split; [ exact Hvalid | ]. split; [ apply grows_refl | ].
  split; [ intros x [] | ].
  split.
  - intros index numbers use statics valuation valuation' Hin _ _ _ v
      (c & n & Hv & Hkind & Hnumber & Hn).
    exists c, n. repeat split; try assumption. simpl.
    destruct Hin as [[= <- <-] | Hin].
    + apply fold_union_keeps. exact Hn.
    + exact (fold_union_in _ _ index numbers n Hin Hn).
  - intros Hk. split; [ exact Hk | intros simple kind [] ].
Qed.

Section Reduce.

Context
  (reduce_head_of_kind_naked_immediate :
     Typing_env.t -> head_of_kind_naked_immediate
     -> Or_bottom.t head_of_kind_naked_immediate).

Hypothesis Hreduce :
  reduce_head_of_kind_naked_immediate_sound
    reduce_head_of_kind_naked_immediate.

(* The numbers that the fold answers hold those that it was given to
   start with, and every number that has one of the heads. *)
Lemma join_naked_immediates_spec (s : state)
    (heads : list (Index.t * head_of_kind_naked_immediate)) :
  ∀ (immediates : Or_unknown.t (list Z)) (imms : list Z),
    join_naked_immediates joined_env reduce_head_of_kind_naked_immediate s
      immediates heads
    = Some (Or_unknown.Known imms) ->
    (∃ imms0 : list Z,
       immediates = Or_unknown.Known imms0 ∧ ∀ n : Z, n ∈ imms0 -> n ∈ imms)
    ∧ ∀ (index : Index.t) (head : head_of_kind_naked_immediate)
        (use : Typing_env.t) (statics : static_env) (valuation : env)
        (v : value),
        In (index, head) heads ->
        joined_env s index = Some use ->
        typing_env_satisfied_strictly statics valuation use ->
        head_of_kind_naked_immediate_satisfied statics valuation v head ->
        ∃ n : Z, v = Value_const (Naked_immediate n) ∧ n ∈ imms.
Proof.
  induction heads as [ | [id2 head2] heads IH]; intros immediates imms Hjoin;
    simpl in Hjoin.
  - injection Hjoin as ->. split; [ exists imms; auto | intros ? ? ? ? ? ? [] ].
  - destruct (joined_env s id2) as [tenv | ] eqn:Htenv; [ | discriminate ].
    destruct (reduce_head_of_kind_naked_immediate tenv head2)
      as [head2' | ] eqn:Hreduced.
    + destruct (IH _ imms Hjoin) as ((imms0' & Himms0' & Hkeeps) & Hheads).
      destruct immediates as [imms0 | ]; [ | discriminate ].
      destruct (naked_immediates head2') as [imms2 | ] eqn:Himms2;
        [ | discriminate ].
      injection Himms0' as <-.
      split.
      * exists imms0. split; [ reflexivity | ].
        intros n Hn. apply Hkeeps, numbers_union_in. left. exact Hn.
      * intros index head use statics valuation v [[= <- <-] | Hin] Huse
          Hsatisfied Hhead.
        -- rewrite Htenv in Huse. injection Huse as <-.
           destruct (Hreduce tenv statics valuation v head2 Hsatisfied Hhead)
             as (head' & Hhead' & n & Hv & Hn & _).
           rewrite Hreduced in Hhead'. injection Hhead' as <-.
           rewrite Himms2 in Hn.
           exists n. split; [ exact Hv | ].
           apply Hkeeps, numbers_union_in. right. exact Hn.
        -- exact (Hheads index head use statics valuation v Hin Huse
                    Hsatisfied Hhead).
    + destruct (IH _ imms Hjoin) as (Himms0 & Hheads).
      split; [ exact Himms0 | ].
      intros index head use statics valuation v [[= <- <-] | Hin] Huse
        Hsatisfied Hhead.
      * rewrite Htenv in Huse. injection Huse as <-.
        destruct (Hreduce tenv statics valuation v head2 Hsatisfied Hhead)
          as (head' & Hhead' & _).
        congruence.
      * exact (Hheads index head use statics valuation v Hin Huse Hsatisfied
                 Hhead).
Qed.

Theorem n_way_join_head_of_kind_naked_immediate_is_sound :
  n_way_join_head_of_kind_naked_immediate_sound typing_env_well_formed
    joined_env definitions_of_existentials kind_in_target_env state_valid
    state_kinded reduce_head_of_kind_naked_immediate.
Proof.
  intros s s' heads answer Hvalid Hwf Hkinded Hdistinct Hargs Hstep.
  unfold n_way_join_head_of_kind_naked_immediate in Hstep.
  destruct
    (join_naked_immediates joined_env reduce_head_of_kind_naked_immediate s
       (Or_unknown.Known []) heads)
    as [[imms | ] | ] eqn:Hjoin; [ | | discriminate ].
  - cbv beta iota in Hstep.
    destruct (Keyed.is_empty imms); [ discriminate | ].
    injection Hstep as <- <-.
    destruct (join_naked_immediates_spec s heads _ _ Hjoin) as (_ & Hheads).
    split; [ exact Hvalid | ]. split; [ apply grows_refl | ].
    split; [ intros x [] | ].
    split.
    + intros index head use statics valuation valuation' Hin Huse Hsatisfied
        _ v Hhead.
      destruct (Hheads index head use statics valuation v Hin Huse Hsatisfied
                  Hhead)
        as (n & Hv & Hn).
      exists n. split; [ exact Hv | ]. split; [ exact Hn | intros ? ? [] ].
    + intros Hk. split; [ exact Hk | intros simple kind [] ].
  - injection Hstep as <- <-.
    split; [ exact Hvalid | ]. split; [ apply grows_refl | ].
    split; [ intros x [] | ].
    split; [ intros; exact I | ].
    intros Hk. split; [ exact Hk | intros simple kind [] ].
Qed.

End Reduce.

(* ======================================================================== *)
(* Environment extensions                                                   *)
(* ======================================================================== *)

Section Extension.

Context
  (join_env_n_way_join_env_extension :
     state -> list (Index.t * env_extension)
     -> option (Or_bottom.t (env_extension * state))).

Hypothesis Hextension :
  n_way_join_env_extension_sound typing_env_well_formed joined_env
    definitions_of_existentials kind_in_target_env state_valid state_kinded
    join_env_n_way_join_env_extension.

Theorem n_way_join_env_extension_is_sound :
  n_way_join_env_extension_sound_here typing_env_well_formed joined_env
    definitions_of_existentials kind_in_target_env state_valid state_kinded
    join_env_n_way_join_env_extension.
Proof.
  intros s s' extensions answer Hvalid Hwf Hkinded Hdistinct Hargs Hstep.
  unfold n_way_join_env_extension in Hstep.
  destruct (join_env_n_way_join_env_extension s extensions)
    as [joined | ] eqn:Hjoin; [ | discriminate ].
  assert (Hscope :
    args_in_scope env_extension_in_scope s extensions).
  { intros index extension Hin.
    destruct (Hargs index extension Hin) as (use & Huse & Hknown & _).
    exists use. split; [ exact Huse | exact Hknown ]. }
  assert (Hkinds :
    args_in_scope env_extension_kinded s extensions).
  { intros index extension Hin.
    destruct (Hargs index extension Hin) as (use & Huse & _ & Haliases).
    exists use. split; [ exact Huse | exact Haliases ]. }
  destruct
    (Hextension s extensions joined Hvalid Hwf Hkinded Hdistinct Hscope Hjoin)
    as (Hbottom & Hok & Hkinded').
  destruct joined as [[joined s1] | ].
  - injection Hstep as <- <-.
    destruct (Hok joined s1 eq_refl) as (Hvalid1 & Hgrows & Htarget & Hbound).
    split; [ exact Hvalid1 | ]. split; [ exact Hgrows | ].
    split; [ exact Htarget | ].
    split.
    + intros index extension use statics valuation valuation' Hin Huse
        Hsatisfied Hextends Hholds.
      exact
        (Hbound index extension use statics valuation valuation' Hin Huse
           Hsatisfied Hextends Hholds).
    + intros Hk. exact (Hkinded' Hk Hkinds joined s1 eq_refl).
  - injection Hstep as <- <-. rewrite (Hbottom eq_refl).
    split; [ exact Hvalid | ]. split; [ apply grows_refl | ].
    split; [ intros x [] | ].
    split; [ intros ? ? ? ? ? ? [] | ].
    intros Hk. split; [ exact Hk | intros simple kind [] ].
Qed.

End Extension.

(* ======================================================================== *)
(* The functions that call the join of types                                *)
(* ======================================================================== *)

Section Functions.

Context
  (n_way_join_simples :
     state -> flambda_kind -> list (Index.t * simple)
     -> option (Or_bottom.t simple * state))
  (import_type : state -> flambda_type -> option state)
  (join_env_n_way_join_env_extension :
     state -> list (Index.t * env_extension)
     -> option (Or_bottom.t (env_extension * state)))
  (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
  (reduce_head_of_kind_naked_immediate :
     Typing_env.t -> head_of_kind_naked_immediate
     -> Or_bottom.t head_of_kind_naked_immediate)
  (type_eqb : flambda_type -> flambda_type -> bool)
  (block_shape_eqb : block_shape -> block_shape -> bool).

Hypothesis Hsimples :
  n_way_join_simples_sound typing_env_well_formed joined_env
    definitions_of_existentials kind_in_target_env state_valid state_kinded
    n_way_join_simples.
Hypothesis Himport :
  import_type_sound typing_env_well_formed joined_env
    definitions_of_existentials kind_in_target_env state_valid state_kinded
    import_type.
Hypothesis Hconstants :
  joined_envs_constants_of_kind typing_env_well_formed joined_env state_valid.
Hypothesis Hexpand : expand_head_sound typing_env_well_formed expand_head.
Hypothesis Hreduce :
  reduce_head_of_kind_naked_immediate_sound
    reduce_head_of_kind_naked_immediate.
Hypothesis Heqb : type_eqb_sound type_eqb.
Hypothesis Hshape : block_shape_eqb_sound block_shape_eqb.

(* The lists at which the things given and the answers are kinded
   ([kinds_with]): the aliases, to which the names of the equations
   and the checks of well-formedness are added or not. *)
Variable with_equations : bool.
Variable with_checks : bool.

Local Abbreviation kinds := (kinds_with with_equations with_checks).
Local Abbreviation kinds_of_head :=
  (kinds_of_head_of_kind_value with_equations with_checks).
Local Abbreviation kinds_of_non_null :=
  (kinds_of_head_of_kind_value_non_null with_equations with_checks).
Local Abbreviation kinds_of_blocks :=
  (kinds_of_row_like_for_blocks with_equations with_checks).
Local Abbreviation kinds_of_closures :=
  (kinds_of_row_like_for_closures with_equations with_checks).
Local Abbreviation kinds_of_entry :=
  (kinds_of_closures_entry with_equations with_checks).
Local Abbreviation kinds_of_extension :=
  (kinds_of_env_extension with_equations with_checks).
Local Abbreviation kinds_of_type_at :=
  (kinds_at with_equations with_checks).
Local Abbreviation kinds_of_immediates :=
  (kinds_of_type_at (Naked_number Kind_naked_immediate)).
Local Abbreviation checked := (check with_checks).

(* [ill_formed] is of no kind in the target. *)
Lemma ill_formed_of_no_kind_in_target (s : state) :
  ¬ of_kind_in_target s (fst ill_formed) (snd ill_formed).
Proof. intros H. cbn in H. discriminate H. Qed.

(* A check that an argument has passes; and a check of the answer
   that passes asks nothing. *)
Lemma arg_check_passes {X : Type} (variables_of : X -> list variable)
    (kinds_of : X -> list (simple * flambda_kind)) (s : state)
    (args : list (Index.t * X)) (index : Index.t) (x : X) (test : bool) :
  args_in_scope (arg_ok variables_of kinds_of) s args ->
  In (index, x) args ->
  incl (checked test) (kinds_of x) ->
  with_checks = true ->
  test = true.
Proof.
  intros Hargs Hin Hincl Hchecks.
  destruct (Hargs index x Hin) as (use & _ & _ & Hkinds).
  destruct test; [ reflexivity | ]. exfalso.
  apply (ill_formed_of_no_kind use). apply Hkinds, Hincl.
  unfold check. rewrite Hchecks. left. reflexivity.
Qed.

Lemma answer_check_passes (s : state) (test : bool) (simple : simple)
    (kind : flambda_kind) :
  (with_checks = true -> test = true) ->
  (simple, kind) ∈ checked test ->
  of_kind_in_target s simple kind.
Proof.
  intros Htest Hin. apply in_check in Hin as (Hchecks & Hfalse & _).
  rewrite (Htest Hchecks) in Hfalse. discriminate Hfalse.
Qed.

(* A type is kinded in an environment, at those lists. *)
Definition kinded_with (tenv : Typing_env.t) (ty : flambda_type) : Prop :=
  ∀ (simple : simple) (kind : flambda_kind),
    (simple, kind) ∈ kinds ty -> simple_of_kind tenv simple kind.

Lemma kinded_with_aliases (tenv : Typing_env.t) (ty : flambda_type) :
  kinded_with tenv ty -> type_kinded tenv ty.
Proof.
  exact (kinds_ask_aliases (simple_of_kind tenv) with_equations with_checks
           ty (ill_formed_of_no_kind tenv)).
Qed.

(* What is supposed at those lists of the import of a type, of the
   join of extensions and of the expansion of a head. With nothing
   added it is in the contracts above; with the names of the
   equations, it is those and the contracts of the names; with the
   checks, those and what is supposed for well-formedness. *)
Hypothesis Himport_kinds :
  ∀ (s s' : state) (ty : flambda_type),
    state_valid s -> envs_well_formed s -> envs_kinded s ->
    import_type s ty = Some s' ->
    state_kinded s ->
    (∃ (index : Index.t) (use : Typing_env.t),
       joined_env s index = Some use ∧ kinded_with use ty) ->
    state_kinded s' ∧ kinded_in_target kinds s' ty.
Hypothesis Hextension_kinds :
  step_sound free_variables_of_env_extension kinds_of_extension
    free_variables_of_env_extension kinds_of_extension
    (fun statics valuation extension valuation' joined =>
       env_extension_satisfied statics valuation extension ->
       env_extension_satisfied statics valuation' joined)
    (n_way_join_env_extension join_env_n_way_join_env_extension).
Hypothesis Hexpand_kinds :
  ∀ (tenv : Typing_env.t) (ty : flambda_type) (expanded : Expanded_type.t),
    expand_head tenv ty = Some expanded ->
    typing_env_well_formed tenv ->
    typing_env_kinded tenv ->
    kinded_with tenv ty ->
    kinded_with tenv (Expanded_type.to_type expanded).

(* The contract of the join of types ([n_way_join_type_sound]), at
   those lists. *)
Definition type_join_sound
    (n_way_join :
       state -> list (Index.t * flambda_type)
       -> option (Or_unknown.t flambda_type * state)) : Prop :=
  ∀ (s s' : state) (ts : list (Index.t * flambda_type))
    (result : Or_unknown.t flambda_type),
    state_valid s ->
    envs_well_formed s ->
    envs_kinded s ->
    join_args_distinct ts ->
    args_in_scope type_in_scope s ts ->
    args_in_scope kinded_with s ts ->
    n_way_join s ts = Some (result, s') ->
    state_valid s'
    ∧ grows s s'
    ∧ (∀ ty : flambda_type,
         result = Or_unknown.Known ty ->
         type_in_target kind_in_target_env s' ty
         ∧ (∀ (index : Index.t) (t_index : flambda_type),
              In (index, t_index) ts -> kind ty = kind t_index)
         ∧ ∀ (index : Index.t) (t_index : flambda_type)
             (use : Typing_env.t) (statics : static_env)
             (valuation valuation' : env) (v : value),
             In (index, t_index) ts ->
             joined_env s' index = Some use ->
             typing_env_satisfied_strictly statics valuation use ->
             extends s' index valuation valuation' ->
             type_satisfied statics valuation v t_index ->
             type_satisfied statics valuation' v ty)
    ∧ (state_kinded s ->
       state_kinded s'
       ∧ ∀ ty : flambda_type,
           result = Or_unknown.Known ty -> kinded_in_target kinds s' ty).

(* The contract of the join of types is that of a sound function. *)
Lemma type_sound_step
    (n_way_join :
       state -> list (Index.t * flambda_type)
       -> option (Or_unknown.t flambda_type * state)) :
  type_join_sound n_way_join ->
  step_sound free_variables kinds (of_known free_variables) (of_known kinds)
    type_bound n_way_join.
Proof.
  intros Hsound s s' ts answer Hvalid Hwf Hkinded Hdistinct Hargs Hstep.
  assert (Hscope : args_in_scope type_in_scope s ts).
  { intros index t Hin. destruct (Hargs index t Hin) as (use & Huse & H & _).
    exists use. split; [ exact Huse | exact H ]. }
  assert (Hkinds : args_in_scope kinded_with s ts).
  { intros index t Hin. destruct (Hargs index t Hin) as (use & Huse & _ & H).
    exists use. split; [ exact Huse | exact H ]. }
  destruct
    (Hsound s s' ts answer Hvalid Hwf Hkinded Hdistinct Hscope Hkinds Hstep)
    as (Hvalid' & Hgrows & Hknown & Hkinded').
  split; [ exact Hvalid' | ]. split; [ exact Hgrows | ].
  split; [ | split ].
  - destruct answer as [ty | ]; [ | intros x [] ].
    exact (proj1 (Hknown ty eq_refl)).
  - intros index t use statics valuation valuation' Hin Huse Hsatisfied
      Hextends.
    destruct answer as [ty | ]; [ | exact I ].
    intros v Hv.
    exact
      (proj2 (proj2 (Hknown ty eq_refl)) index t use statics valuation
         valuation' v Hin Huse Hsatisfied Hextends Hv).
  - intros Hk. destruct (Hkinded' Hk) as (Hk' & Hty).
    split; [ exact Hk' | ].
    destruct answer as [ty | ]; [ | intros simple kind [] ].
    exact (Hty ty eq_refl).
Qed.

(* What is proved of the functions that call the join of types, of a
   join of types [n_way_join_rec] that is sound. *)
Section Body.

Variable n_way_join_rec :
  state -> list (Index.t * flambda_type)
  -> option (Or_unknown.t flambda_type * state).

Hypothesis Hrec_sound : type_join_sound n_way_join_rec.
Hypothesis Hrec_nil : ∀ s : state, n_way_join_rec s [] = None.

Lemma Hrec :
  step_sound free_variables kinds (of_known free_variables)
    (of_known kinds) type_bound n_way_join_rec.
Proof. exact (type_sound_step _ Hrec_sound). Qed.

(* The same where a kind is asked of the types: the type answered is
   of the kind of the types given. *)
Lemma Hrec_at (k : flambda_kind) :
  step_sound free_variables (kinds_of_type_at k) (of_known free_variables)
    (of_known (kinds_of_type_at k)) type_bound n_way_join_rec.
Proof.
  intros s s' ts answer Hvalid Hwf Hkinded Hdistinct Hargs Hstep.
  assert (Hargs' : args_in_scope (arg_ok free_variables kinds) s ts).
  { apply (args_of_parts _ _ _ _ s ts ts Hargs). intros index t Hin.
    exists t. split; [ exact Hin | ].
    split; [ apply incl_refl | apply incl_appr, incl_refl ]. }
  destruct (Hrec s s' ts answer Hvalid Hwf Hkinded Hdistinct Hargs' Hstep)
    as (Hvalid' & Hgrows & Htarget & Hbound & Hkinds).
  split; [ exact Hvalid' | ]. split; [ exact Hgrows | ].
  split; [ exact Htarget | ]. split; [ exact Hbound | ].
  intros Hk. destruct (Hkinds Hk) as [Hk' Hanswer]. split; [ exact Hk' | ].
  destruct answer as [ty | ]; [ | intros simple kind [] ].
  intros simple k' Hin. cbn [of_known] in Hin. unfold kinds_at in Hin.
  apply in_app_or in Hin as [Hin | Hin]; [ | exact (Hanswer simple k' Hin) ].
  apply (answer_check_passes s' (kind_eqb (kind ty) k) simple k');
    [ | exact Hin ].
  intros Hchecks.
  destruct ts as [ | [index t] ts'];
    [ rewrite Hrec_nil in Hstep; discriminate | ].
  assert (Hscope : args_in_scope type_in_scope s ((index, t) :: ts')).
  { intros index' t' Hin'.
    destruct (Hargs' index' t' Hin') as (use & Huse & H & _).
    exists use. split; [ exact Huse | exact H ]. }
  assert (Hkinds_ts : args_in_scope kinded_with s ((index, t) :: ts')).
  { intros index' t' Hin'.
    destruct (Hargs' index' t' Hin') as (use & Huse & _ & H).
    exists use. split; [ exact Huse | exact H ]. }
  destruct
    (Hrec_sound s s' ((index, t) :: ts') (Or_unknown.Known ty) Hvalid Hwf
       Hkinded Hdistinct Hscope Hkinds_ts Hstep)
    as (_ & _ & Hknown & _).
  rewrite (proj1 (proj2 (Hknown ty eq_refl)) index t (or_introl eq_refl)).
  exact (arg_check_passes _ _ s ((index, t) :: ts') index t _ Hargs
           (or_introl eq_refl) (incl_appl _ (incl_refl _)) Hchecks).
Qed.

(* The invariant of a fold that threads the state from [s]. *)
Definition threaded (s s1 : state) : Prop :=
  state_valid s1 ∧ grows s s1 ∧ (state_kinded s -> state_kinded s1).

(* ------------------------------------------------------------------------ *)
(* Products indexed by slots                                                *)
(* ------------------------------------------------------------------------ *)

Definition variables_of_product (product : finite_map flambda_type)
    : list variable :=
  flat_map (fun binding => free_variables (snd binding)) product.

Definition aliases_of_product (product : finite_map flambda_type)
    : list (simple * flambda_kind) :=
  flat_map (fun binding => kinds (snd binding)) product.

(* Each component of the join bounds the component of each product
   given. *)
Definition product_bound (statics : static_env) (valuation : env)
    (product : finite_map flambda_type) (valuation' : env)
    (joined : finite_map flambda_type) : Prop :=
  ∀ (g : positive) (ty : flambda_type),
    lookup g joined = Some ty ->
    ∃ ty_i : flambda_type,
      lookup g product = Some ty_i
      ∧ ∀ v : value,
          type_satisfied statics valuation v ty_i ->
          type_satisfied statics valuation' v ty.

Lemma lookup_in_product {A : Type} (g : positive) (a : A)
    (product : finite_map A) :
  lookup g product = Some a -> In (g, a) product.
Proof. apply (keyed_lookup_in Pos.eqb Pos.eqb_eq). Qed.

Theorem n_way_join_generic_product_is_sound :
  step_sound variables_of_product aliases_of_product variables_of_product
    aliases_of_product product_bound
    (n_way_join_generic_product n_way_join_rec).
Proof.
  intros s s' products answer Hvalid Hwf Hkinded Hdistinct Hargs Hstep.
  unfold n_way_join_generic_product in Hstep.
  destruct products as [ | product0 products0] eqn:Hproducts.
  { injection Hstep as <- <-.
    split; [ exact Hvalid | ]. split; [ apply grows_refl | ].
    split; [ intros x [] | ]. split; [ intros ? ? ? ? ? ? [] | ].
    intros Hk. split; [ exact Hk | intros simple kind [] ]. }
  rewrite <- Hproducts in *. clear Hproducts product0 products0.
  (* The types gathered for an index are arguments of the join. *)
  assert (Hgathered :
    ∀ (g : positive) (tys : list (Index.t * flambda_type)),
      lookup g (gather_components products) = Some tys ->
      join_args_distinct tys
      ∧ args_in_scope (arg_ok free_variables kinds) s tys
      ∧ ∀ (index : Index.t) (product : finite_map flambda_type),
          In (index, product) products ->
          ∃ ty_i : flambda_type,
            lookup g product = Some ty_i ∧ In (index, ty_i) tys).
  { intros g tys Htys.
    destruct (gather_components_spec products g tys Htys)
      as (Hfst & Hto & Hfrom).
    split; [ | split; [ | exact Hto ] ].
    - unfold join_args_distinct. rewrite Hfst. apply NoDup_rev.
      exact Hdistinct.
    - apply (args_of_parts _ _ _ _ s products tys Hargs).
      intros index ty Hin.
      destruct (Hfrom index ty Hin) as (product & Hty & Hin').
      exists product. split; [ exact Hin' | ].
      apply lookup_in_product in Hty.
      split; intros y Hy;
        [ unfold variables_of_product | unfold aliases_of_product ];
        apply in_flat_map; exists (g, ty); split; assumption. }
  (* Each step of the fold is a call of the join of types. *)
  set (step :=
    fun (_ : positive) (tys : list (Index.t * flambda_type)) (env : state) =>
      match n_way_join_rec env tys with
      | Some (joined, env) =>
        match joined with
        | Or_unknown.Known ty => Some (Some ty, env)
        | Or_unknown.Unknown => Some (None, env)
        end
      | None => None
      end) in Hstep.
  assert (Hcall :
    ∀ (g : positive) (tys : list (Index.t * flambda_type)) (s1 : state)
      (a : option flambda_type) (s2 : state),
      lookup g (gather_components products) = Some tys ->
      threaded s s1 ->
      step g tys s1 = Some (a, s2) ->
      threaded s s2 ∧ grows s1 s2
      ∧ ∀ ty : flambda_type,
          a = Some ty ->
          answer_holds_from (of_known free_variables)
            (of_known kinds) type_bound s1 s2 tys
            (Or_unknown.Known ty)).
  { intros g tys s1 a s2 Htys (Hvalid1 & Hgrows1 & Hk1) Hstep1.
    destruct (Hgathered g tys Htys) as (Hdistinct' & Hargs' & _).
    unfold step in Hstep1.
    destruct (n_way_join_rec s1 tys) as [[joined s2'] | ] eqn:Hjoin;
      [ | discriminate ].
    destruct
      (step_sound_at _ _ _ _ _ _ Hrec s s1 s2' tys joined Hwf Hkinded Hgrows1
         Hvalid1 Hdistinct' Hargs' Hjoin)
      as (Hvalid2 & Hgrows2 & Hk2 & Hholds).
    assert (Hs2 : s2' = s2) by (destruct joined; congruence). subst s2'.
    split; [ | split; [ exact Hgrows2 | ] ].
    - split; [ exact Hvalid2 | ].
      split; [ exact (grows_trans _ _ _ Hgrows1 Hgrows2) | ].
      intros Hk. exact (Hk2 (Hk1 Hk)).
    - intros ty ->. destruct joined as [ty' | ]; [ | discriminate ].
      injection Hstep1 as ->. exact Hholds. }
  destruct
    (filter_map_fold_spec Pos.eqb Pos.eqb_eq (threaded s) grows grows_refl
       grows_trans step (gather_components products)
       (fun g tys s1 a s2 Htys Hs1 Hstep1 =>
          let H := Hcall g tys s1 a s2 Htys Hs1 Hstep1 in
          conj (proj1 H) (proj1 (proj2 H)))
       s answer s'
       (conj Hvalid (conj (grows_refl s) (fun H => H))) Hstep)
    as ((Hvalid' & Hgrows & Hk') & _ & Hin & Hfound & _).
  (* What a binding of the answer comes from. *)
  assert (Hbinding :
    ∀ (g : positive) (ty : flambda_type),
      lookup g answer = Some ty ->
      ∃ (tys : list (Index.t * flambda_type)) (s1 s2 : state),
        lookup g (gather_components products) = Some tys
        ∧ grows s2 s' ∧ (state_kinded s -> state_kinded s1)
        ∧ answer_holds_from (of_known free_variables)
            (of_known kinds) type_bound s1 s2 tys
            (Or_unknown.Known ty)).
  { intros g ty Hty.
    destruct (Hfound g ty Hty)
      as (tys & s1 & s2 & Htys & Hs1 & _ & Hstep1 & Hgrows2).
    exists tys, s1, s2. split; [ exact Htys | ]. split; [ exact Hgrows2 | ].
    split; [ exact (proj2 (proj2 Hs1)) | ].
    exact (proj2 (proj2 (Hcall g tys s1 _ s2 Htys Hs1 Hstep1)) ty eq_refl). }
  split; [ exact Hvalid' | ]. split; [ exact Hgrows | ].
  split; [ | split ].
  - intros x Hx. apply in_flat_map in Hx as ([g ty] & Hin' & Hx).
    destruct (Hbinding g ty (Hin g ty Hin'))
      as (tys & s1 & s2 & _ & Hgrows2 & _ & Hholds).
    exact (proj1 (Hholds s' Hgrows2) x Hx).
  - intros index product use statics valuation valuation' Hin' Huse
      Hsatisfied Hextends g ty Hty.
    destruct (Hbinding g ty Hty)
      as (tys & s1 & s2 & Htys & Hgrows2 & _ & Hholds).
    destruct (Hgathered g tys Htys) as (_ & _ & Hto).
    destruct (Hto index product Hin') as (ty_i & Hty_i & Hin'').
    exists ty_i. split; [ exact Hty_i | ].
    exact
      (proj1 (proj2 (Hholds s' Hgrows2)) index ty_i use statics valuation
         valuation' Hin'' Huse Hsatisfied Hextends).
  - intros Hk. split; [ exact (Hk' Hk) | ].
    intros simple kind Hsimple.
    apply in_flat_map in Hsimple as ([g ty] & Hin' & Hsimple).
    destruct (Hbinding g ty (Hin g ty Hin'))
      as (tys & s1 & s2 & _ & Hgrows2 & Hk1 & Hholds).
    exact (proj2 (proj2 (Hholds s' Hgrows2)) (Hk1 Hk) simple kind Hsimple).
Qed.

(* ------------------------------------------------------------------------ *)
(* Products indexed by integers: the fields of blocks                       *)
(* ------------------------------------------------------------------------ *)

Definition variables_of_fields (tys : list flambda_type) : list variable :=
  flat_map free_variables tys.

Definition aliases_of_fields (tys : list flambda_type)
    : list (simple * flambda_kind) :=
  flat_map kinds tys.

(* The fields of a block of the shape that have the types given have
   the types joined. *)
Definition fields_bound (shape : block_shape) (statics : static_env)
    (valuation : env) (tys : list flambda_type) (valuation' : env)
    (joined : list flambda_type) : Prop :=
  ∀ (tag : Z) (fields : list value),
    block_has_shape shape tag fields ->
    fields_satisfied (type_satisfied statics valuation) tys fields ->
    fields_satisfied (type_satisfied statics valuation') joined fields.

(* A value that has a type in an input, whose environment knows the
   variables of the type, has it under a valuation that extends the
   input's. *)
Lemma type_satisfied_extends (s : state) (index : Index.t) (use : Typing_env.t)
    (statics : static_env) (valuation valuation' : env) (v : value)
    (ty : flambda_type) :
  joined_env s index = Some use ->
  (∀ x : variable, x ∈ free_variables ty -> knows use x) ->
  extends s index valuation valuation' ->
  type_satisfied statics valuation v ty ->
  type_satisfied statics valuation' v ty.
Proof.
  intros Huse Hknown (use' & Huse' & Hagrees & _) Hty.
  rewrite Huse in Huse'. injection Huse' as <-.
  apply (type_satisfied_coincide statics valuation valuation' v ty);
    [ | exact Hty ].
  intros x Hx. symmetry. apply Hagrees, Hknown, Hx.
Qed.

(* [import_types]: every type given is in the target at every later
   state, and kinded there if it was kinded in an input. *)
Lemma import_types_spec (tys : list flambda_type) :
  ∀ s s' : state,
    state_valid s -> envs_well_formed s -> envs_kinded s ->
    import_types import_type s tys = Some s' ->
    state_valid s' ∧ grows s s'
    ∧ (∀ ty : flambda_type,
         ty ∈ tys -> in_target free_variables s' ty)
    ∧ (state_kinded s ->
       (∀ ty : flambda_type,
          ty ∈ tys ->
          ∃ (index : Index.t) (use : Typing_env.t),
            joined_env s index = Some use ∧ kinded_with use ty) ->
       state_kinded s'
       ∧ ∀ ty : flambda_type,
           ty ∈ tys -> kinded_in_target kinds s' ty).
Proof.
  induction tys as [ | ty tys IH]; intros s s' Hvalid Hwf Hkinded Himports;
    simpl in Himports.
  - injection Himports as <-.
    split; [ exact Hvalid | ]. split; [ apply grows_refl | ].
    split; [ intros ? [] | ].
    intros Hk _. split; [ exact Hk | intros ? [] ].
  - destruct (import_type s ty) as [s1 | ] eqn:Himported; [ | discriminate ].
    destruct (Himport s s1 ty Hvalid Hwf Hkinded Himported)
      as (Hvalid1 & Hgrows1 & Htarget1 & Hkinds1).
    destruct
      (IH s1 s' Hvalid1 (grows_well_formed _ _ Hgrows1 Hwf)
         (grows_kinded _ _ Hgrows1 Hkinded) Himports)
      as (Hvalid' & Hgrows' & Htargets & Hkinds').
    split; [ exact Hvalid' | ].
    split; [ exact (grows_trans _ _ _ Hgrows1 Hgrows') | ].
    split.
    + intros ty' [<- | Hin].
      * exact (grows_in_target free_variables _ _ ty Hgrows' Htarget1).
      * exact (Htargets ty' Hin).
    + intros Hk Hkinded_tys.
      destruct
        (Himport_kinds s s1 ty Hvalid Hwf Hkinded Himported Hk
           (Hkinded_tys ty (or_introl eq_refl)))
        as (Hk1 & Hty).
      assert (Hrest :
        ∀ ty' : flambda_type,
          ty' ∈ tys ->
          ∃ (index : Index.t) (use : Typing_env.t),
            joined_env s1 index = Some use ∧ kinded_with use ty').
      { intros ty' Hin.
        destruct (Hkinded_tys ty' (or_intror Hin)) as (index & use & Huse & H).
        exists index, use. rewrite (grows_joined_env _ _ index Hgrows1).
        split; assumption. }
      destruct (Hkinds' Hk1 Hrest) as (Hk' & Htys).
      split; [ exact Hk' | ].
      intros ty' [<- | Hin].
      * exact (grows_kinded_in_target kinds _ _ ty Hgrows' Hty).
      * exact (Htys ty' Hin).
Qed.

(* Fields that are the same have the same values. *)
Lemma fields_the_same_spec (least_length : nat) :
  ∀ first_fields other_fields : list flambda_type,
    fields_the_same type_eqb least_length first_fields other_fields = true ->
    ∀ i : nat,
      i < least_length ->
      ∃ first_field other_field : flambda_type,
        nth_error first_fields i = Some first_field
        ∧ nth_error other_fields i = Some other_field
        ∧ ∀ (statics : static_env) (vars : env) (v : value),
            type_satisfied statics vars v other_field
            <-> type_satisfied statics vars v first_field.
Proof.
  induction least_length as [ | least_length IH];
    intros first_fields other_fields Hsame i Hi; [ lia | ].
  destruct first_fields as [ | first_field first_fields];
    [ discriminate | ].
  destruct other_fields as [ | other_field other_fields];
    [ discriminate | ].
  simpl in Hsame. apply andb_true_iff in Hsame as [Hfield Hsame].
  destruct i as [ | i].
  - exists first_field, other_field. repeat split; try reflexivity.
    + apply (Heqb other_field first_field Hfield).
    + apply (Heqb other_field first_field Hfield).
  - apply (IH first_fields other_fields Hsame i). lia.
Qed.

(* The slow path: the fields joined from an index on. *)
Lemma join_fields_spec (s : state) (shape : block_shape)
    (fields : list (Index.t * list flambda_type)) :
  envs_well_formed s -> envs_kinded s ->
  join_args_distinct fields ->
  args_in_scope (arg_ok variables_of_fields aliases_of_fields) s fields ->
  ∀ (count index : nat) (s1 : state) (answer : list flambda_type)
    (s' : state),
    threaded s s1 ->
    join_fields n_way_join_rec s1 shape fields index count
    = Some (answer, s') ->
    threaded s s' ∧ grows s1 s'
    ∧ in_target variables_of_fields s' answer
    ∧ (∀ (k : nat) (ty : flambda_type),
         nth_error answer k = Some ty ->
         ∀ (id : Index.t) (tys : list flambda_type) (use : Typing_env.t)
           (statics : static_env) (valuation valuation' : env),
           In (id, tys) fields ->
           joined_env s' id = Some use ->
           typing_env_satisfied_strictly statics valuation use ->
           extends s' id valuation valuation' ->
           ∃ ty_i : flambda_type,
             nth_error tys (index + k) = Some ty_i
             ∧ ∀ field : value,
                 (∀ kind : flambda_kind,
                    block_shape_element_kind shape (index + k) = Some kind ->
                    kind_of_value field = kind) ->
                 type_satisfied statics valuation field ty_i ->
                 type_satisfied statics valuation' field ty)
    ∧ (state_kinded s -> kinded_in_target aliases_of_fields s' answer).
Proof.
  intros Hwf Hkinded Hdistinct Hargs.
  induction count as [ | count IH]; intros index s1 answer s' Hs1 Hjoin;
    simpl in Hjoin.
  - injection Hjoin as <- <-.
    split; [ exact Hs1 | ]. split; [ apply grows_refl | ].
    split; [ intros x [] | ].
    split; [ intros [ | k] ty [=] | intros _ simple kind [] ].
  - destruct (fields_at index fields) as [tys | ] eqn:Htys; [ | discriminate ].
    destruct (n_way_join_rec s1 tys) as [[joined s2] | ] eqn:Hjoined;
      [ | discriminate ].
    destruct
      (match joined with
       | Or_unknown.Known ty => Some ty
       | Or_unknown.Unknown => unknown_from_shape shape index
       end) as [ty0 | ] eqn:Hty0; [ | discriminate ].
    destruct (join_fields n_way_join_rec s2 shape fields (S index) count)
      as [[rest s3] | ] eqn:Hrest; [ | discriminate ].
    injection Hjoin as <- <-.
    rewrite fields_at_eq in Htys.
    destruct (indexed_map_opt_spec _ _ tys Htys) as (Hfst & Hto & Hfrom).
    destruct Hs1 as (Hvalid1 & Hgrows1 & Hk1).
    assert (Hdistinct' : join_args_distinct tys).
    { unfold join_args_distinct. rewrite Hfst. exact Hdistinct. }
    assert (Hargs' :
      args_in_scope (arg_ok free_variables kinds) s tys).
    { apply (args_of_parts _ _ _ _ s fields tys Hargs).
      intros id ty Hin. destruct (Hfrom id ty Hin) as (tys0 & Hnth & Hin').
      exists tys0. split; [ exact Hin' | ]. apply nth_error_In in Hnth.
      split; intros y Hy;
        [ unfold variables_of_fields | unfold aliases_of_fields ];
        apply in_flat_map; exists ty; split; assumption. }
    destruct
      (step_sound_at _ _ _ _ _ _ Hrec s s1 s2 tys joined Hwf Hkinded Hgrows1
         Hvalid1 Hdistinct' Hargs' Hjoined)
      as (Hvalid2 & Hgrows2 & Hk2 & Hholds).
    assert (Hs2 : threaded s s2).
    { split; [ exact Hvalid2 | ].
      split; [ exact (grows_trans _ _ _ Hgrows1 Hgrows2) | ].
      intros Hk. exact (Hk2 (Hk1 Hk)). }
    destruct (IH (S index) s2 rest s3 Hs2 Hrest)
      as (Hs3 & Hgrows3 & Htarget & Hbound & Hkinds).
    destruct (Hholds s3 Hgrows3) as (Htarget0 & Hbound0 & Hkinds0).
    split; [ exact Hs3 | ].
    split; [ exact (grows_trans _ _ _ Hgrows2 Hgrows3) | ].
    split; [ | split ].
    + intros x Hx. unfold variables_of_fields in Hx. simpl in Hx.
      apply in_app_or in Hx as [Hx | Hx]; [ | exact (Htarget x Hx) ].
      destruct joined as [ty | ].
      * injection Hty0 as <-. exact (Htarget0 x Hx).
      * unfold unknown_from_shape in Hty0.
        destruct (block_shape_element_kind shape index) as [kind | ];
          [ | discriminate ].
        injection Hty0 as <-. rewrite free_variables_unknown in Hx.
        destruct Hx.
    + intros [ | k] ty Hty id tys0 use statics valuation valuation' Hin Huse
        Hsatisfied Hextends.
      * injection Hty as <-. rewrite Nat.add_0_r.
        destruct (Hto id tys0 Hin) as (ty_i & Hnth & Hin').
        exists ty_i. split; [ exact Hnth | ].
        intros field Hkind Hfield.
        destruct joined as [ty | ].
        -- injection Hty0 as <-.
           exact
             (Hbound0 id ty_i use statics valuation valuation' Hin' Huse
                Hsatisfied Hextends field Hfield).
        -- unfold unknown_from_shape in Hty0.
           destruct (block_shape_element_kind shape index) as [kind | ]
             eqn:Hkind'; [ | discriminate ].
           injection Hty0 as <-. apply unknown_satisfied.
           exact (Hkind kind eq_refl).
      * rewrite <- Nat.add_succ_comm.
        exact
          (Hbound k ty Hty id tys0 use statics valuation valuation' Hin Huse
             Hsatisfied Hextends).
    + intros Hk simple kind Hsimple. unfold aliases_of_fields in Hsimple.
      simpl in Hsimple.
      apply in_app_or in Hsimple as [Hsimple | Hsimple];
        [ | exact (Hkinds Hk simple kind Hsimple) ].
      destruct joined as [ty | ].
      * injection Hty0 as <-. exact (Hkinds0 (Hk1 Hk) simple kind Hsimple).
      * unfold unknown_from_shape in Hty0.
        destruct (block_shape_element_kind shape index) as [kind' | ];
          [ | discriminate ].
        injection Hty0 as <-. rewrite kinds_with_unknown in Hsimple.
        destruct Hsimple.
Qed.

Theorem n_way_join_int_indexed_product_is_sound (shape : block_shape) :
  step_sound variables_of_fields aliases_of_fields variables_of_fields
    aliases_of_fields (fields_bound shape)
    (fun s fields =>
       n_way_join_int_indexed_product import_type type_eqb n_way_join_rec s
         shape fields).
Proof.
  intros s s' fields answer Hvalid Hwf Hkinded Hdistinct Hargs Hstep.
  unfold n_way_join_int_indexed_product in Hstep.
  destruct fields as [ | [id1 first_fields] other_fields] eqn:Hfields;
    [ discriminate | ].
  rewrite <- Hfields in *.
  set (least_length :=
    List.fold_left
      (fun least_length other_fields =>
         Nat.min least_length (length (snd other_fields)))
      other_fields (length first_fields)) in Hstep.
  destruct (least_length_le other_fields (length first_fields))
    as (Hfirst & Hothers).
  fold least_length in Hfirst, Hothers.
  assert (Hleast :
    ∀ (index : Index.t) (tys : list flambda_type),
      In (index, tys) fields -> least_length <= length tys).
  { intros index tys Hin. rewrite Hfields in Hin.
    destruct Hin as [[= <- <-] | Hin];
      [ exact Hfirst | exact (Hothers index tys Hin) ]. }
  destruct
    (forallb
       (fun other_fields =>
          fields_the_same type_eqb least_length first_fields
            (snd other_fields))
       other_fields) eqn:Hsame.
  - (* The fields are the same: those of one input, imported. *)
    destruct
      (find
         (fun fields_of_id =>
            Nat.eqb (length (snd fields_of_id)) least_length)
         fields) as [[id_j fields_j] | ] eqn:Hfind; [ | discriminate ].
    apply find_some in Hfind as [Hin_j Hlength_j]. simpl in Hlength_j.
    apply Nat.eqb_eq in Hlength_j.
    destruct (import_types import_type s fields_j) as [s1 | ] eqn:Himports;
      [ | discriminate ].
    injection Hstep as <- <-.
    destruct (import_types_spec fields_j s s1 Hvalid Hwf Hkinded Himports)
      as (Hvalid1 & Hgrows1 & Htargets & Hkinds).
    (* Every input has, below the least length, fields with the values
       of the first's. *)
    assert (Hsame_first :
      ∀ (index : Index.t) (tys : list flambda_type) (i : nat),
        In (index, tys) fields ->
        i < least_length ->
        ∃ first_field ty_i : flambda_type,
          nth_error first_fields i = Some first_field
          ∧ nth_error tys i = Some ty_i
          ∧ ∀ (statics : static_env) (vars : env) (v : value),
              type_satisfied statics vars v ty_i
              <-> type_satisfied statics vars v first_field).
    { intros index tys i Hin Hi. rewrite Hfields in Hin.
      destruct Hin as [[= <- <-] | Hin].
      - destruct (nth_error first_fields i) as [first_field | ] eqn:Hnth.
        + exists first_field, first_field. repeat split; auto.
        + apply nth_error_None in Hnth. lia.
      - rewrite forallb_forall in Hsame.
        exact
          (fields_the_same_spec least_length first_fields tys
             (Hsame (index, tys) Hin) i Hi). }
    split; [ exact Hvalid1 | ]. split; [ exact Hgrows1 | ].
    split; [ | split ].
    + intros x Hx. apply in_flat_map in Hx as (ty & Hin & Hx).
      exact (Htargets ty Hin x Hx).
    + intros index tys use statics valuation valuation' Hin Huse Hsatisfied
        Hextends tag vals _ Htys.
      rewrite fields_satisfied_nth_iff in Htys.
      apply fields_satisfied_nth_iff. intros i ty_j field Hty_j Hfield.
      assert (Hi : i < least_length).
      { rewrite <- Hlength_j. apply nth_error_Some. congruence. }
      destruct (Hsame_first index tys i Hin Hi)
        as (first_field & ty_i & Hfirst_field & Hty_i & Hi_first).
      destruct (Hsame_first id_j fields_j i Hin_j Hi)
        as (first_field' & ty_j' & Hfirst_field' & Hty_j' & Hj_first).
      rewrite Hfirst_field in Hfirst_field'. injection Hfirst_field' as <-.
      rewrite Hty_j in Hty_j'. injection Hty_j' as <-.
      apply Hj_first, Hi_first.
      destruct (Hargs index tys Hin) as (use0 & Huse0 & Hknown & _).
      rewrite <- (grows_joined_env _ _ index Hgrows1), Huse in Huse0.
      injection Huse0 as <-.
      apply (type_satisfied_extends s1 index use statics valuation valuation'
               field ty_i Huse);
        [ | exact Hextends | exact (Htys i ty_i field Hty_i Hfield) ].
      intros x Hx. apply Hknown. apply in_flat_map. exists ty_i.
      split; [ exact (nth_error_In _ _ Hty_i) | exact Hx ].
    + intros Hk.
      assert (Hkinded_j :
        ∀ ty : flambda_type,
          ty ∈ fields_j ->
          ∃ (index : Index.t) (use : Typing_env.t),
            joined_env s index = Some use ∧ kinded_with use ty).
      { intros ty Hin.
        destruct (Hargs id_j fields_j Hin_j) as (use & Huse & _ & Haliases).
        exists id_j, use. split; [ exact Huse | ].
        intros simple kind Hsimple. apply Haliases. apply in_flat_map.
        exists ty. split; assumption. }
      destruct (Hkinds Hk Hkinded_j) as (Hk1 & Htys).
      split; [ exact Hk1 | ].
      intros simple kind Hsimple.
      apply in_flat_map in Hsimple as (ty & Hin & Hsimple).
      exact (Htys ty Hin simple kind Hsimple).
  - (* The fields are joined one by one. *)
    destruct
      (join_fields_spec s shape fields Hwf Hkinded Hdistinct Hargs
         least_length 0 s answer s'
         (conj Hvalid (conj (grows_refl s) (fun H => H))) Hstep)
      as ((Hvalid' & Hgrows & Hk') & _ & Htarget & Hbound & Hkinds).
    split; [ exact Hvalid' | ]. split; [ exact Hgrows | ].
    split; [ exact Htarget | ].
    split.
    + intros index tys use statics valuation valuation' Hin Huse Hsatisfied
        Hextends tag vals Hshape' Htys.
      rewrite fields_satisfied_nth_iff in Htys.
      apply fields_satisfied_nth_iff. intros k ty field Hty Hfield.
      destruct
        (Hbound k ty Hty index tys use statics valuation valuation' Hin Huse
           Hsatisfied Hextends)
        as (ty_i & Hty_i & Hkept).
      simpl in Hty_i. apply Hkept.
      * intros kind Hkind.
        exact
          (block_shape_element_kind_of_field shape tag vals k field kind
             Hshape' Hfield Hkind).
      * exact (Htys k ty_i field Hty_i Hfield).
    + intros Hk. split; [ exact (Hk' Hk) | exact (Hkinds Hk) ].
Qed.

(* ------------------------------------------------------------------------ *)
(* Closures entries                                                         *)
(* ------------------------------------------------------------------------ *)

(* A closure with its set of closures: what has a closures entry. *)
Definition closure_and_set : Type :=
  value * function_slot * list (function_slot * code_id)
  * list (value_slot * value).

Definition closures_entry_holds (statics : static_env) (vars : env)
    (w : closure_and_set) (entry : closures_entry) : Prop :=
  let '(v, slot, functions, value_slots) := w in
  closures_entry_satisfied statics vars v slot functions value_slots entry.

Lemma rev_map_distinct {A B : Type} (f : A -> B)
    (entries : list (Index.t * A)) :
  join_args_distinct entries ->
  join_args_distinct
    (rev (map (fun entry => (fst entry, f (snd entry))) entries)).
Proof.
  unfold join_args_distinct. intros Hdistinct.
  rewrite map_rev, map_map. simpl. apply NoDup_rev. exact Hdistinct.
Qed.

Lemma rev_map_in {A B : Type} (f : A -> B) (entries : list (Index.t * A))
    (index : Index.t) (b : B) :
  In (index, b) (rev (map (fun entry => (fst entry, f (snd entry))) entries))
  <-> ∃ a : A, In (index, a) entries ∧ b = f a.
Proof.
  rewrite <- in_rev, in_map_iff. split.
  - intros ([index' a] & [= <- <-] & Hin). exists a. split; auto.
  - intros (a & Hin & ->). exists (index, a). split; auto.
Qed.

Theorem n_way_join_closures_entry_is_sound :
  step_sound free_variables_of_closures_entry kinds_of_entry
    free_variables_of_closures_entry kinds_of_entry
    (all_kept closures_entry_holds closures_entry_holds)
    (n_way_join_closures_entry n_way_join_rec).
Proof.
  intros s s' entries answer Hvalid Hwf Hkinded Hdistinct Hargs Hstep.
  unfold n_way_join_closures_entry,
    n_way_join_function_slot_indexed_product,
    n_way_join_value_slot_indexed_product in Hstep.
  set (closure_types :=
    rev
      (map
         (fun entry => (fst entry, closures_entry_closure_types (snd entry)))
         entries)) in Hstep.
  set (value_slot_types :=
    rev
      (map
         (fun entry =>
            (fst entry, closures_entry_value_slot_types (snd entry)))
         entries)) in Hstep.
  destruct (n_way_join_generic_product n_way_join_rec s closure_types)
    as [[closure_types' s1] | ] eqn:Hclosures; [ | discriminate ].
  match type of Hstep with
  | context [for_all_bindings_b ?p ?m] =>
    destruct (for_all_bindings_b p m); [ | discriminate ]
  end.
  destruct (n_way_join_generic_product n_way_join_rec s1 value_slot_types)
    as [[value_slot_types' s2] | ] eqn:Hvalue_slots; [ | discriminate ].
  injection Hstep as <- <-.
  assert (Hargs1 :
    args_in_scope (arg_ok variables_of_product aliases_of_product) s
      closure_types).
  { apply (args_of_parts _ _ _ _ s entries closure_types Hargs).
    intros index product Hin. apply rev_map_in in Hin as (entry & Hin & ->).
    exists entry. split; [ exact Hin | ].
    destruct entry as [function_types closures value_slots]. simpl.
    split; apply incl_appl, incl_refl. }
  assert (Hargs2 :
    args_in_scope (arg_ok variables_of_product aliases_of_product) s
      value_slot_types).
  { apply (args_of_parts _ _ _ _ s entries value_slot_types Hargs).
    intros index product Hin. apply rev_map_in in Hin as (entry & Hin & ->).
    exists entry. split; [ exact Hin | ].
    destruct entry as [function_types closures value_slots]. simpl.
    split; apply incl_appr, incl_refl. }
  destruct
    (step_sound_at _ _ _ _ _ _ n_way_join_generic_product_is_sound s s s1
       closure_types closure_types' Hwf Hkinded (grows_refl s) Hvalid
       (rev_map_distinct _ entries Hdistinct) Hargs1 Hclosures)
    as (Hvalid1 & Hgrows1 & Hk1 & Hholds1).
  destruct
    (step_sound_at _ _ _ _ _ _ n_way_join_generic_product_is_sound s s1 s2
       value_slot_types value_slot_types' Hwf Hkinded Hgrows1 Hvalid1
       (rev_map_distinct _ entries Hdistinct) Hargs2 Hvalue_slots)
    as (Hvalid2 & Hgrows2 & Hk2 & Hholds2).
  destruct (Hholds1 s2 Hgrows2) as (Htarget1 & Hbound1 & Hkinds1).
  destruct (Hholds2 s2 (grows_refl s2)) as (Htarget2 & Hbound2 & Hkinds2).
  split; [ exact Hvalid2 | ].
  split; [ exact (grows_trans _ _ _ Hgrows1 Hgrows2) | ].
  split; [ | split ].
  - intros x Hx. simpl in Hx.
    apply in_app_or in Hx as [Hx | Hx];
      [ exact (Htarget1 x Hx) | exact (Htarget2 x Hx) ].
  - intros index [function_types closures value_slots] use statics valuation
      valuation' Hin Huse Hsatisfied Hextends [[[v slot] functions] slots]
      Hentry.
    unfold closures_entry_holds in *.
    apply
      (n_way_join_function_types_sound statics valuation' v slot functions
         slots _ index function_types).
    { apply in_map_iff.
      exists (index, Closures_entry function_types closures value_slots).
      split; [ reflexivity | exact Hin ]. }
    rewrite closures_entry_lookup_iff in Hentry.
    destruct Hentry as (Hfunctions & Hclosures' & Hslots).
    apply closures_entry_lookup_iff.
    split; [ exact Hfunctions | ].
    split.
    + intros g ty closure Hty Hclosure.
      assert (Hin1 : In (index, closures) closure_types).
      { apply rev_map_in.
        exists (Closures_entry function_types closures value_slots).
        split; [ exact Hin | reflexivity ]. }
      destruct
        (Hbound1 index closures use statics valuation valuation' Hin1 Huse
           Hsatisfied Hextends g ty Hty)
        as (ty_i & Hty_i & Hkept).
      exact (Hkept closure (Hclosures' g ty_i closure Hty_i Hclosure)).
    + intros w ty contents Hty Hcontents.
      assert (Hin2 : In (index, value_slots) value_slot_types).
      { apply rev_map_in.
        exists (Closures_entry function_types closures value_slots).
        split; [ exact Hin | reflexivity ]. }
      destruct
        (Hbound2 index value_slots use statics valuation valuation' Hin2 Huse
           Hsatisfied Hextends w ty Hty)
        as (ty_i & Hty_i & Hkept).
      exact (Hkept contents (Hslots w ty_i contents Hty_i Hcontents)).
  - intros Hk. split; [ exact (Hk2 (Hk1 Hk)) | ].
    intros simple kind Hsimple. simpl in Hsimple.
    apply in_app_or in Hsimple as [Hsimple | Hsimple].
    + exact (Hkinds1 Hk simple kind Hsimple).
    + exact (Hkinds2 (Hk1 Hk) simple kind Hsimple).
Qed.

(* ------------------------------------------------------------------------ *)
(* Cases of row-like structures                                             *)
(* ------------------------------------------------------------------------ *)

Section Cases.

(* [W] is what a case admits: a tag and fields, or a closure with its
   set. *)
Context {lattice shape maps_to W : Type}
  (n_way_join_maps_to :
     state -> shape -> list (Index.t * maps_to) -> option (maps_to * state))
  (equal_index : lattice -> lattice -> bool)
  (inter_index : lattice -> lattice -> lattice)
  (n_way_join_shape : list shape -> or_unknown_or_bottom shape)
  (variables_of_maps_to : maps_to -> list variable)
  (aliases_of_maps_to : maps_to -> list (simple * flambda_kind))
  (index_holds : W -> row_like_index lattice shape -> Prop)
  (maps_to_holds : static_env -> env -> W -> maps_to -> Prop).

Local Abbreviation a_case :=
  (row_like_case lattice shape maps_to env_extension).

Definition variables_of_case (c : a_case) : list variable :=
  variables_of_maps_to (case_maps_to c)
  ++ free_variables_of_env_extension (case_env_extension c).

Definition aliases_of_case (c : a_case) : list (simple * flambda_kind) :=
  aliases_of_maps_to (case_maps_to c)
  ++ kinds_of_extension (case_env_extension c).

Definition case_holds (statics : static_env) (vars : env) (w : W) (c : a_case)
    : Prop :=
  index_holds w (case_index c)
  ∧ maps_to_holds statics vars w (case_maps_to c)
  ∧ env_extension_satisfied statics vars (case_env_extension c).

Hypothesis Hindex :
  ∀ (indexes : list (row_like_index lattice shape))
    (index : row_like_index lattice shape),
    n_way_join_index equal_index inter_index n_way_join_shape indexes
    = Ok index ->
    ∀ i : row_like_index lattice shape,
      In i indexes -> ∀ w : W, index_holds w i -> index_holds w index.
Hypothesis Hindex_bottom :
  ∀ indexes : list (row_like_index lattice shape),
    n_way_join_index equal_index inter_index n_way_join_shape indexes
    = Bottom ->
    indexes = [].
Hypothesis Hmaps_to :
  ∀ index : row_like_index lattice shape,
    step_sound variables_of_maps_to aliases_of_maps_to variables_of_maps_to
      aliases_of_maps_to
      (fun statics valuation x valuation' y =>
         ∀ w : W,
           index_holds w index ->
           maps_to_holds statics valuation w x ->
           maps_to_holds statics valuation' w y)
      (fun s args => n_way_join_maps_to s (index_shape index) args).

Theorem n_way_join_case_is_sound :
  case_join_sound variables_of_case aliases_of_case
    (of_known variables_of_case) (of_known aliases_of_case)
    (known_bound (all_kept case_holds case_holds))
    (n_way_join_case join_env_n_way_join_env_extension n_way_join_maps_to
       equal_index inter_index n_way_join_shape).
Proof.
  intros s s1 cases result Hwf Hkinded Hgrows Hvalid Hdistinct Hargs Hjoin.
  unfold n_way_join_case in Hjoin.
  destruct
    (n_way_join_index equal_index inter_index n_way_join_shape
       (map (fun c => case_index (snd c)) cases))
    as [ | index | ] eqn:Hjoined.
  - injection Hjoin as <-.
    split; [ exact Hvalid | ]. split; [ apply grows_refl | ].
    split; [ auto | ].
    intros s' Hgrows'. split; [ intros x [] | ].
    split; [ intros; exact I | intros _ simple kind [] ].
  - set (maps_tos := map (fun c => (fst c, case_maps_to (snd c))) cases)
      in Hjoin.
    set (extensions :=
      map (fun c => (fst c, case_env_extension (snd c))) cases) in Hjoin.
    destruct (n_way_join_maps_to s1 (index_shape index) maps_tos)
      as [[joined_maps_to s2] | ] eqn:Hmaps; [ | discriminate ].
    cbn in Hjoin.
    destruct
      (n_way_join_env_extension join_env_n_way_join_env_extension s2
         extensions)
      as [[extension s3] | ] eqn:Hext; [ | discriminate ].
    cbn in Hjoin. injection Hjoin as <-.
    assert (Hdistinct1 : join_args_distinct maps_tos).
    { unfold join_args_distinct, maps_tos. rewrite map_map. exact Hdistinct. }
    assert (Hdistinct2 : join_args_distinct extensions).
    { unfold join_args_distinct, extensions. rewrite map_map.
      exact Hdistinct. }
    assert (Hargs1 :
      args_in_scope (arg_ok variables_of_maps_to aliases_of_maps_to) s
        maps_tos).
    { apply (args_of_parts _ _ _ _ s cases maps_tos Hargs). intros id m Hin.
      apply in_map_iff in Hin as ([id' c] & [= <- <-] & Hin).
      exists c. split; [ exact Hin | ]. split; apply incl_appl, incl_refl. }
    assert (Hargs2 :
      args_in_scope
        (arg_ok free_variables_of_env_extension kinds_of_extension) s
        extensions).
    { apply (args_of_parts _ _ _ _ s cases extensions Hargs).
      intros id m Hin.
      apply in_map_iff in Hin as ([id' c] & [= <- <-] & Hin).
      exists c. split; [ exact Hin | ]. split; apply incl_appr, incl_refl. }
    destruct
      (step_sound_at _ _ _ _ _ _ (Hmaps_to index) s s1 s2 maps_tos
         joined_maps_to Hwf Hkinded Hgrows Hvalid Hdistinct1 Hargs1 Hmaps)
      as (Hvalid2 & Hgrows2 & Hk2 & Hholds1).
    destruct
      (step_sound_at _ _ _ _ _ _
         Hextension_kinds
         s s2 s3 extensions extension Hwf Hkinded
         (grows_trans _ _ _ Hgrows Hgrows2) Hvalid2 Hdistinct2 Hargs2 Hext)
      as (Hvalid3 & Hgrows3 & Hk3 & Hholds2).
    split; [ exact Hvalid3 | ].
    split; [ exact (grows_trans _ _ _ Hgrows2 Hgrows3) | ].
    split; [ auto | ].
    intros s' Hgrows'.
    destruct (Hholds1 s' (grows_trans _ _ _ Hgrows3 Hgrows'))
      as (Htarget1 & Hbound1 & Hkinds1).
    destruct (Hholds2 s' Hgrows') as (Htarget2 & Hbound2 & Hkinds2).
    split; [ | split ].
    + intros x Hx. cbn [of_known] in Hx. unfold variables_of_case in Hx.
      cbn [case_maps_to case_env_extension] in Hx.
      apply in_app_or in Hx as [Hx | Hx];
        [ exact (Htarget1 x Hx) | exact (Htarget2 x Hx) ].
    + intros id c use statics valuation valuation' Hin Huse Hsatisfied
        Hextends w (Hi & Hm & He).
      assert (Hi' : index_holds w index).
      { apply (Hindex _ index Hjoined (case_index c)); [ | exact Hi ].
        apply in_map_iff. exists (id, c). split; [ reflexivity | exact Hin ]. }
      split; [ exact Hi' | ]. cbn [case_maps_to case_env_extension].
      split.
      * apply
          (Hbound1 id (case_maps_to c) use statics valuation valuation');
          [ | exact Huse | exact Hsatisfied | exact Hextends | exact Hi'
          | exact Hm ].
        apply in_map_iff. exists (id, c). split; [ reflexivity | exact Hin ].
      * apply
          (Hbound2 id (case_env_extension c) use statics valuation
             valuation');
          [ | exact Huse | exact Hsatisfied | exact Hextends | exact He ].
        apply in_map_iff. exists (id, c). split; [ reflexivity | exact Hin ].
    + intros Hk simple kind Hsimple. cbn [of_known] in Hsimple.
      unfold aliases_of_case in Hsimple.
      cbn [case_maps_to case_env_extension] in Hsimple.
      apply in_app_or in Hsimple as [Hsimple | Hsimple];
        [ exact (Hkinds1 Hk simple kind Hsimple)
        | exact (Hkinds2 (Hk2 Hk) simple kind Hsimple) ].
  - injection Hjoin as <-. apply Hindex_bottom in Hjoined.
    exact (map_eq_nil _ _ Hjoined).
Qed.

Theorem n_way_join_knowns_is_sound :
  case_join_sound (of_known variables_of_case) (of_known aliases_of_case)
    (of_known variables_of_case) (of_known aliases_of_case)
    (or_unknown_bound (all_kept case_holds case_holds))
    (n_way_join_knowns join_env_n_way_join_env_extension n_way_join_maps_to
       equal_index inter_index n_way_join_shape).
Proof.
  intros s s1 cases result Hwf Hkinded Hgrows Hvalid Hdistinct Hargs Hjoin.
  unfold n_way_join_knowns in Hjoin. rewrite all_known_eq in Hjoin.
  destruct (indexed_map_opt _ cases) as [known | ] eqn:Hknown.
  - destruct (indexed_map_opt_spec _ _ known Hknown) as (Hfst & Hto & Hfrom).
    assert (Hdistinct' : join_args_distinct known).
    { unfold join_args_distinct. rewrite Hfst. exact Hdistinct. }
    assert (Hargs' :
      args_in_scope (arg_ok variables_of_case aliases_of_case) s known).
    { intros id c Hin. destruct (Hfrom id c Hin) as ([c' | ] & Heq & Hin');
        [ injection Heq as <- | discriminate ].
      exact (Hargs id (Or_unknown.Known c') Hin'). }
    pose proof
      (n_way_join_case_is_sound s s1 known result Hwf Hkinded Hgrows Hvalid
         Hdistinct' Hargs' Hjoin)
      as Hcase.
    destruct result as [[answer s2] | ].
    + destruct Hcase as (Hvalid2 & Hgrows2 & Hk2 & Hholds).
      split; [ exact Hvalid2 | ]. split; [ exact Hgrows2 | ].
      split; [ exact Hk2 | ].
      intros s' Hgrows'.
      destruct (Hholds s' Hgrows') as (Htarget & Hbound & Hkinds).
      split; [ exact Htarget | ]. split; [ | exact Hkinds ].
      intros id x use statics valuation valuation' Hin Huse Hsatisfied
        Hextends.
      destruct (Hto id x Hin) as (c & Hc & Hin').
      destruct x as [c' | ]; [ injection Hc as -> | discriminate ].
      specialize
        (Hbound id c use statics valuation valuation' Hin' Huse Hsatisfied
           Hextends).
      destruct answer; exact Hbound.
    + subst known. symmetry in Hfst. exact (map_eq_nil _ _ Hfst).
  - injection Hjoin as <-.
    split; [ exact Hvalid | ]. split; [ apply grows_refl | ].
    split; [ auto | ].
    intros s' _. split; [ intros x [] | ].
    split; [ | intros _ simple kind [] ].
    intros id x use statics valuation valuation' _ _ _ _. destruct x; exact I.
Qed.

End Cases.

(* ------------------------------------------------------------------------ *)
(* Row-like structures of blocks                                            *)
(* ------------------------------------------------------------------------ *)

(* A block: its tag and its fields. *)
Definition tag_and_fields : Type := (Z * list value)%type.

Definition block_index_holds (w : tag_and_fields)
    (index : row_like_index nat block_shape) : Prop :=
  block_index_satisfied (fst w) (snd w) index.

Definition block_fields_hold (statics : static_env) (vars : env)
    (w : tag_and_fields) (tys : list flambda_type) : Prop :=
  fields_satisfied (type_satisfied statics vars) tys (snd w).

Definition blocks_hold (statics : static_env) (vars : env)
    (w : tag_and_fields) (blocks : row_like_for_blocks) : Prop :=
  row_like_for_blocks_satisfied statics vars (fst w) (snd w) blocks.

Local Abbreviation block_case_variables :=
  (variables_of_case (lattice := nat) (shape := block_shape)
     variables_of_fields).
Local Abbreviation block_case_aliases :=
  (aliases_of_case (lattice := nat) (shape := block_shape) aliases_of_fields).
Local Abbreviation block_case_holds :=
  (case_holds block_index_holds block_fields_hold).

Lemma block_shapes_sound (shapes : list block_shape) (joined : block_shape) :
  n_way_join_block_shape block_shape_eqb shapes = Ok joined ->
  ∀ shape' : block_shape, In shape' shapes -> shape' = joined.
Proof.
  destruct shapes as [ | shape1 rest]; [ discriminate | ]. simpl.
  destruct (forallb (block_shape_eqb shape1) rest) eqn:Hall;
    [ | discriminate ].
  intros [= <-] shape' [<- | Hin]; [ reflexivity | ].
  rewrite forallb_forall in Hall. symmetry.
  exact (Hshape _ _ (Hall shape' Hin)).
Qed.

Lemma block_index_sound (indexes : list (row_like_index nat block_shape))
    (index : row_like_index nat block_shape) :
  n_way_join_index Nat.eqb Nat.min (n_way_join_block_shape block_shape_eqb)
    indexes
  = Ok index ->
  ∀ i : row_like_index nat block_shape,
    In i indexes ->
    ∀ w : tag_and_fields, block_index_holds w i -> block_index_holds w index.
Proof.
  intros Hjoined i Hin w [Hdomain Hshape'].
  refine
    (n_way_join_index_sound Nat.eqb Nat.min
       (n_way_join_block_shape block_shape_eqb)
       (fun (w : tag_and_fields) size => length (snd w) = size)
       (fun (w : tag_and_fields) size => size <= length (snd w))
       (fun (w : tag_and_fields) shape =>
          block_has_shape shape (fst w) (snd w))
       _ _ _ _ _ indexes index Hjoined i Hin w Hdomain Hshape').
  - intros w' l <-. apply Nat.le_refl.
  - intros w' l l' Heq Hl. apply Nat.eqb_eq in Heq. congruence.
  - intros w' l l' Hl. exact (Nat.le_trans _ _ _ (Nat.le_min_l l l') Hl).
  - intros w' l l' Hl. exact (Nat.le_trans _ _ _ (Nat.le_min_r l l') Hl).
  - intros shapes joined Hshapes shape' Hin' w' Hholds.
    rewrite <- (block_shapes_sound shapes joined Hshapes shape' Hin').
    exact Hholds.
Qed.

Lemma block_index_bottom (indexes : list (row_like_index nat block_shape)) :
  n_way_join_index Nat.eqb Nat.min (n_way_join_block_shape block_shape_eqb)
    indexes
  = Bottom ->
  indexes = [].
Proof.
  apply n_way_join_index_bottom. intros [ | shape1 rest]; [ reflexivity | ].
  simpl. destruct (forallb (block_shape_eqb shape1) rest); discriminate.
Qed.

Lemma block_fields_sound (index : row_like_index nat block_shape) :
  step_sound variables_of_fields aliases_of_fields variables_of_fields
    aliases_of_fields
    (fun statics valuation x valuation' y =>
       ∀ w : tag_and_fields,
         block_index_holds w index ->
         block_fields_hold statics valuation w x ->
         block_fields_hold statics valuation' w y)
    (fun s args =>
       n_way_join_int_indexed_product import_type type_eqb n_way_join_rec s
         (index_shape index) args).
Proof.
  apply (step_sound_bound _ _ _ _ (fields_bound (index_shape index)));
    [ | apply n_way_join_int_indexed_product_is_sound ].
  intros statics valuation x valuation' y Hbound w [_ Hshape'] Hfields.
  exact (Hbound (fst w) (snd w) Hshape' Hfields).
Qed.

(* ------------------------------------------------------------------------ *)
(* The shape of the cases of blocks that are answered                       *)
(* ------------------------------------------------------------------------ *)

Lemma join_fields_length (shape : block_shape)
    (fields : list (Index.t * list flambda_type)) (count : nat) :
  ∀ (index : nat) (s s' : state) (tys : list flambda_type),
    join_fields n_way_join_rec s shape fields index count = Some (tys, s') ->
    length tys = count.
Proof.
  induction count as [ | count IH]; intros index s s' tys Hjoin;
    simpl in Hjoin.
  - injection Hjoin as <- _. reflexivity.
  - destruct (fields_at index fields) as [tys0 | ]; simpl in Hjoin;
      [ | discriminate ].
    destruct (n_way_join_rec s tys0) as [[joined s1] | ]; simpl in Hjoin;
      [ | discriminate ].
    destruct joined as [ty | ]; simpl in Hjoin;
      [ | destruct (unknown_from_shape shape index) as [ty | ];
          simpl in Hjoin; [ | discriminate ] ].
    all: destruct (join_fields n_way_join_rec s1 shape fields (S index) count)
           as [[tys1 s2] | ] eqn:Hrest; simpl in Hjoin; [ | discriminate ].
    all: injection Hjoin as <- _; simpl; rewrite (IH _ _ _ _ Hrest);
      reflexivity.
Qed.

(* The fields answered are as many as the shortest list given has. *)
Lemma int_indexed_product_length (shape : block_shape) (s s' : state)
    (id : Index.t) (first : list flambda_type)
    (others : list (Index.t * list flambda_type)) (tys : list flambda_type) :
  n_way_join_int_indexed_product import_type type_eqb n_way_join_rec s shape
    ((id, first) :: others)
  = Some (tys, s') ->
  length tys
  = List.fold_left
      (fun least other => Nat.min least (length (snd other))) others
      (length first).
Proof.
  unfold n_way_join_int_indexed_product. cbv zeta.
  set (least := List.fold_left _ others (length first)).
  destruct (forallb _ others).
  - destruct (find _ _) as [[id' fields'] | ] eqn:Hfind; [ | discriminate ].
    apply find_some in Hfind as [_ Hlength]. cbn [snd] in Hlength.
    apply Nat.eqb_eq in Hlength.
    destruct (import_types import_type s fields') as [s1 | ]; simpl;
      [ | discriminate ].
    intros [= <- _]. exact Hlength.
  - intros Hjoin. exact (join_fields_length shape _ least 0 s s' tys Hjoin).
Qed.

Lemma block_case_fits_size (c : row_like_block_case) :
  block_case_shape_ok c = true ->
  length (case_maps_to c)
  <= match index_domain (case_index c) with
     | Row_like_index_domain.Known size
     | Row_like_index_domain.At_least size => size
     end.
Proof.
  unfold block_case_shape_ok, block_product_fits. intros H.
  apply andb_true_iff in H as [H _].
  destruct (index_domain (case_index c));
    [ apply Nat.eqb_eq in H | apply Nat.leb_le in H ]; lia.
Qed.

(* The shortest product is no longer than the least size. *)
Lemma least_length_fits (cases : list (Index.t * row_like_block_case)) :
  (∀ (id : Index.t) (c : row_like_block_case),
     In (id, c) cases -> block_case_shape_ok c = true) ->
  ∀ length0 size0 : nat,
    length0 <= size0 ->
    List.fold_left (fun least other => Nat.min least (length (snd other)))
      (map (fun case => (fst case, case_maps_to (snd case))) cases) length0
    <= List.fold_left
         (fun i1' i2 =>
            match index_domain i2 with
            | Row_like_index_domain.At_least i2'
            | Row_like_index_domain.Known i2' => Nat.min i1' i2'
            end)
         (map (fun case => case_index (snd case)) cases) size0.
Proof.
  induction cases as [ | [id c] cases IH]; intros Hok length0 size0 Hle; simpl;
    [ exact Hle | ].
  apply IH; [ intros id' c' Hin; exact (Hok id' c' (or_intror Hin)) | ].
  assert (Hc := block_case_fits_size c (Hok id c (or_introl eq_refl))).
  destruct (index_domain (case_index c)); lia.
Qed.

(* Where every size is exact and the same, so is the shortest product. *)
Lemma least_length_exact (cases : list (Index.t * row_like_block_case))
    (size : nat) :
  (∀ (id : Index.t) (c : row_like_block_case),
     In (id, c) cases -> block_case_shape_ok c = true) ->
  forallb
    (fun i2 =>
       match index_domain i2 with
       | Row_like_index_domain.Known i2' => Nat.eqb size i2'
       | Row_like_index_domain.At_least _ => false
       end)
    (map (fun case => case_index (snd case)) cases)
  = true ->
  List.fold_left (fun least other => Nat.min least (length (snd other)))
    (map (fun case => (fst case, case_maps_to (snd case))) cases) size
  = size.
Proof.
  induction cases as [ | [id c] cases IH]; intros Hok Hall; simpl;
    [ reflexivity | ].
  simpl in Hall. apply andb_true_iff in Hall as [Hc Hall].
  assert (Hfits := Hok id c (or_introl eq_refl)).
  unfold block_case_shape_ok, block_product_fits in Hfits.
  apply andb_true_iff in Hfits as [Hfits _].
  destruct (index_domain (case_index c)) as [n | n]; [ | discriminate Hc ].
  apply Nat.eqb_eq in Hc, Hfits. subst n. rewrite Hfits, Nat.min_id.
  apply IH; [ intros id' c' Hin; exact (Hok id' c' (or_intror Hin)) | ].
  exact Hall.
Qed.

(* The case answered for cases whose products fit their indexes, and
   whose shapes are written in the one way, is such a case. *)
Lemma block_case_join_shape (s s' : state)
    (cases : list (Index.t * row_like_block_case))
    (c : row_like_block_case) :
  (∀ (id : Index.t) (c0 : row_like_block_case),
     In (id, c0) cases -> block_case_shape_ok c0 = true) ->
  n_way_join_case join_env_n_way_join_env_extension
    (n_way_join_int_indexed_product import_type type_eqb n_way_join_rec)
    Nat.eqb Nat.min (n_way_join_block_shape block_shape_eqb) s cases
  = Some (Or_bottom.Ok (Or_unknown.Known c, s')) ->
  block_case_shape_ok c = true.
Proof.
  intros Hok Hjoin. unfold n_way_join_case in Hjoin.
  destruct cases as [ | [id1 c1] rest]; [ discriminate Hjoin | ].
  destruct
    (n_way_join_index Nat.eqb Nat.min (n_way_join_block_shape block_shape_eqb)
       (map (fun case => case_index (snd case)) ((id1, c1) :: rest)))
    as [ | index | ] eqn:Hindex; try discriminate.
  destruct
    (n_way_join_int_indexed_product import_type type_eqb n_way_join_rec s
       (index_shape index)
       (map (fun case => (fst case, case_maps_to (snd case)))
          ((id1, c1) :: rest)))
    as [[m s1] | ] eqn:Hmaps; [ | discriminate ].
  cbn in Hjoin.
  destruct (n_way_join_env_extension join_env_n_way_join_env_extension s1 _)
    as [[extension s2] | ]; [ | discriminate ].
  cbn in Hjoin. injection Hjoin as <- _.
  cbn [map fst snd] in Hmaps, Hindex.
  apply int_indexed_product_length in Hmaps.
  assert (Hok1 := Hok id1 c1 (or_introl eq_refl)).
  assert (Hok_rest :
    ∀ (id : Index.t) (c0 : row_like_block_case),
      In (id, c0) rest -> block_case_shape_ok c0 = true).
  { intros id c0 Hin. exact (Hok id c0 (or_intror Hin)). }
  assert (Hsize1 := block_case_fits_size c1 Hok1).
  unfold block_case_shape_ok in Hok1 |- *.
  apply andb_true_iff in Hok1 as [Hfits1 Hcanonical1].
  cbn [case_index case_maps_to].
  unfold n_way_join_index, n_way_join_block_shape in Hindex.
  cbn [map] in Hindex.
  destruct (forallb (block_shape_eqb (index_shape (case_index c1))) _);
    [ | discriminate Hindex ].
  unfold block_product_fits in Hfits1 |- *.
  assert (Hat_least :
    Nat.leb (length m)
      (List.fold_left
         (fun i1' i2 =>
            match index_domain i2 with
            | Row_like_index_domain.At_least i2'
            | Row_like_index_domain.Known i2' => Nat.min i1' i2'
            end)
         (map (fun case => case_index (snd case)) rest)
         match index_domain (case_index c1) with
         | Row_like_index_domain.Known size
         | Row_like_index_domain.At_least size => size
         end)
    = true).
  { apply Nat.leb_le. rewrite Hmaps.
    exact (least_length_fits rest Hok_rest _ _ Hsize1). }
  destruct (index_domain (case_index c1)) as [size1 | size1] eqn:Hdomain1.
  - destruct (forallb _ (map (fun case => case_index (snd case)) rest))
      eqn:Hall; injection Hindex as <-; cbn [index_domain index_shape].
    + rewrite Hcanonical1, andb_true_r. apply Nat.eqb_eq.
      apply Nat.eqb_eq in Hfits1. rewrite Hmaps, Hfits1.
      exact (least_length_exact rest size1 Hok_rest Hall).
    + rewrite Hat_least, Hcanonical1. reflexivity.
  - injection Hindex as <-. cbn [index_domain index_shape].
    rewrite Hat_least, Hcanonical1. reflexivity.
Qed.

Lemma block_knowns_join_shape (s s' : state)
    (cases : list (Index.t * Or_unknown.t row_like_block_case))
    (v : Or_unknown.t row_like_block_case) :
  (∀ (id : Index.t) (v0 : Or_unknown.t row_like_block_case),
     In (id, v0) cases ->
     match v0 with
     | Or_unknown.Known c0 => block_case_shape_ok c0 = true
     | Or_unknown.Unknown => True
     end) ->
  n_way_join_knowns join_env_n_way_join_env_extension
    (n_way_join_int_indexed_product import_type type_eqb n_way_join_rec)
    Nat.eqb Nat.min (n_way_join_block_shape block_shape_eqb) s cases
  = Some (Or_bottom.Ok (v, s')) ->
  match v with
  | Or_unknown.Known c => block_case_shape_ok c = true
  | Or_unknown.Unknown => True
  end.
Proof.
  intros Hok Hjoin. unfold n_way_join_knowns in Hjoin.
  rewrite all_known_eq in Hjoin.
  match type of Hjoin with
  | context [indexed_map_opt ?f cases] =>
    destruct (indexed_map_opt f cases) as [known | ] eqn:Hknown
  end.
  - destruct v as [c | ]; [ | exact I ].
    apply (block_case_join_shape s s' known c); [ | exact Hjoin ].
    intros id c0 Hin.
    destruct (indexed_map_opt_spec _ _ known Hknown) as (_ & _ & Hfrom).
    destruct (Hfrom id c0 Hin) as ([c' | ] & Heq & Hin');
      [ injection Heq as <- | discriminate ].
    exact (Hok id (Or_unknown.Known c') Hin').
  - injection Hjoin as <- _. exact I.
Qed.

(* The cases of a row-like structure of blocks that is in scope and
   kinded are. *)
Lemma blocks_parts_in_scope (use : Typing_env.t)
    (blocks : row_like_for_blocks) :
  arg_ok free_variables_of_row_like_for_blocks kinds_of_blocks
    use blocks ->
  (∀ (key : Z) (v : Or_unknown.t row_like_block_case),
     Keyed.lookup Z.eqb key (row_like_known_tags blocks) = Some v ->
     arg_ok (of_known block_case_variables) (of_known block_case_aliases) use
       v)
  ∧ ∀ o : row_like_block_case,
      row_like_other_tags blocks = Or_bottom.Ok o ->
      arg_ok block_case_variables block_case_aliases use o.
Proof.
  destruct blocks as [known_tags other_tags alloc_mode].
  intros [Hvariables Haliases]. simpl in *. split.
  - intros key v Hv.
    apply (keyed_lookup_in Z.eqb z_eqb_decides_equality) in Hv.
    split.
    + intros y Hy. apply Hvariables. apply in_or_app. left.
      apply in_flat_map. exists (key, v). split; [ exact Hv | ].
      destruct v; exact Hy.
    + intros simple kind Hsimple. apply Haliases. apply in_or_app. right.
      apply in_or_app. left.
      apply in_flat_map. exists (key, v). split; [ exact Hv | ].
      destruct v; exact Hsimple.
  - intros o ->. split.
    + intros y Hy. apply Hvariables. apply in_or_app. right. exact Hy.
    + intros simple kind Hsimple. apply Haliases. apply in_or_app. right.
      apply in_or_app. right. exact Hsimple.
Qed.

(* The row-like structures that are not bottom, as the fold of
   [n_way_join_row_like_for_blocks] splits them. *)
Definition non_bottom_blocks (row_likes : list (Index.t * row_like_for_blocks))
    : list (Index.t * row_like_for_blocks) :=
  filter
    (fun row_like => negb (row_like_for_blocks_is_bottom (snd row_like)))
    row_likes.

Definition known_of_blocks (row_like : Index.t * row_like_for_blocks)
    : Index.t
      * (Keyed.finite_map Z (Or_unknown.t row_like_block_case)
         * Or_bottom.t row_like_block_case) :=
  (fst row_like,
   (row_like_known_tags (snd row_like), row_like_other_tags (snd row_like))).

Definition other_of_blocks (row_like : Index.t * row_like_for_blocks)
    : Index.t * Or_bottom.t row_like_block_case :=
  (fst row_like, row_like_other_tags (snd row_like)).

Local Abbreviation split_of_blocks :=
  (Or_bottom.t
     (list
        (Index.t
         * (Keyed.finite_map Z (Or_unknown.t row_like_block_case)
            * Or_bottom.t row_like_block_case))
      * list (Index.t * Or_bottom.t row_like_block_case)
      * Alloc_mode.t)).

Definition split_step (acc : split_of_blocks)
    (row_like : Index.t * row_like_for_blocks) : split_of_blocks :=
  if row_like_for_blocks_is_bottom (snd row_like)
  then acc
  else
    match acc with
    | Or_bottom.Bottom =>
      Or_bottom.Ok
        ([known_of_blocks row_like], [other_of_blocks row_like],
         row_like_alloc_mode (snd row_like))
    | Or_bottom.Ok (known, other, alloc_mode) =>
      Or_bottom.Ok
        (known_of_blocks row_like :: known, other_of_blocks row_like :: other,
         join_alloc_mode alloc_mode (row_like_alloc_mode (snd row_like)))
    end.

Definition split_is (kept : list (Index.t * row_like_for_blocks))
    (acc : split_of_blocks) : Prop :=
  match acc with
  | Or_bottom.Bottom => kept = []
  | Or_bottom.Ok (known, other, _) =>
    kept <> []
    ∧ known = rev (map known_of_blocks kept)
    ∧ other = rev (map other_of_blocks kept)
  end.

Lemma split_fold (row_likes : list (Index.t * row_like_for_blocks)) :
  ∀ (kept : list (Index.t * row_like_for_blocks)) (acc : split_of_blocks),
    split_is kept acc ->
    split_is (kept ++ non_bottom_blocks row_likes)
      (List.fold_left split_step row_likes acc).
Proof.
  induction row_likes as [ | row_like row_likes IH]; intros kept acc Hacc.
  - simpl. rewrite app_nil_r. exact Hacc.
  - cbn [List.fold_left].
    assert (Hnext :
      split_is (kept ++ non_bottom_blocks [row_like])
        (split_step acc row_like)).
    { unfold non_bottom_blocks, split_step. cbn [filter].
      destruct (row_like_for_blocks_is_bottom (snd row_like)); cbn [negb].
      - rewrite app_nil_r. exact Hacc.
      - destruct acc as [[[known other] alloc_mode] | ]; simpl in Hacc |- *.
        + destruct Hacc as (Hne & -> & ->).
          split; [ intros Hnil; apply app_eq_nil in Hnil as [_ [=]] | ].
          rewrite !map_app, !rev_app_distr. split; reflexivity.
        + subst kept. split; [ discriminate | split; reflexivity ]. }
    specialize (IH _ _ Hnext).
    replace (kept ++ non_bottom_blocks (row_like :: row_likes))
      with ((kept ++ non_bottom_blocks [row_like])
            ++ non_bottom_blocks row_likes);
      [ exact IH | ].
    rewrite <- app_assoc. f_equal. unfold non_bottom_blocks. cbn [filter].
    destruct (negb _); reflexivity.
Qed.

Theorem n_way_join_row_like_for_blocks_is_sound :
  step_sound free_variables_of_row_like_for_blocks
    kinds_of_blocks
    (of_known free_variables_of_row_like_for_blocks)
    (of_known kinds_of_blocks)
    (known_bound (all_kept blocks_hold blocks_hold))
    (n_way_join_row_like_for_blocks import_type
       join_env_n_way_join_env_extension type_eqb block_shape_eqb
       n_way_join_rec).
Proof.
  intros s s' row_likes answer Hvalid Hwf Hkinded Hdistinct Hargs Hstep.
  unfold n_way_join_row_like_for_blocks in Hstep.
  set (split := List.fold_left _ row_likes Or_bottom.Bottom) in Hstep.
  assert (Hsplit : split_is (non_bottom_blocks row_likes) split)
    by exact (split_fold row_likes [] Or_bottom.Bottom eq_refl).
  assert (Hkept :
    ∀ (index : Index.t) (blocks : row_like_for_blocks),
      In (index, blocks) (non_bottom_blocks row_likes)
      <-> In (index, blocks) row_likes
          ∧ row_like_for_blocks_is_bottom blocks = false).
  { intros index blocks. unfold non_bottom_blocks. rewrite filter_In.
    cbn [snd]. rewrite negb_true_iff. reflexivity. }
  cbv zeta in Hstep.
  destruct split as [[[known other] alloc_mode] | ]; cbn [split_is] in Hsplit.
  2: { injection Hstep as <- <-.
       split; [ exact Hvalid | ]. split; [ apply grows_refl | ].
       split; [ intros x [] | ].
       split.
       - intros index blocks use statics valuation valuation' Hin _ _ _ w Hw.
         exfalso.
         destruct (row_like_for_blocks_is_bottom blocks) eqn:Hb.
         + exact
             (row_like_for_blocks_is_bottom_satisfied statics valuation
                (fst w) (snd w) blocks Hb Hw).
         + assert (Hin' : In (index, blocks) (non_bottom_blocks row_likes))
             by (apply Hkept; split; assumption).
           rewrite Hsplit in Hin'. destruct Hin'.
       - intros Hk. split; [ exact Hk | ]. intros simple kind Hin.
         simpl in Hin. apply in_app_or in Hin as [Hin | Hin];
           [ | destruct Hin ].
         exact (answer_check_passes s _ simple kind (fun _ => eq_refl) Hin). }
  destruct Hsplit as (Hne & -> & ->).
  set (kept := non_bottom_blocks row_likes) in *.
  set (knowns := rev (map known_of_blocks kept)) in *.
  set (others := rev (map other_of_blocks kept)) in *.
  assert (Hkept_distinct : NoDup (map fst kept))
    by (apply filter_indexed_distinct; exact Hdistinct).
  assert (Hknowns_distinct : join_args_distinct knowns).
  { unfold join_args_distinct, knowns. rewrite map_rev, map_map.
    apply NoDup_rev. exact Hkept_distinct. }
  assert (Hothers_distinct : NoDup (map fst others)).
  { unfold others. rewrite map_rev, map_map. apply NoDup_rev.
    exact Hkept_distinct. }
  assert (Hin_knowns :
    ∀ (index : Index.t)
      (known_tags : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
      (other_tags : Or_bottom.t row_like_block_case) (mode : Alloc_mode.t),
      In (index, Row_like_for_blocks known_tags other_tags mode) kept ->
      In (index, (known_tags, other_tags)) knowns).
  { intros index known_tags other_tags mode Hin. unfold knowns.
    rewrite <- in_rev. apply in_map_iff.
    exists (index, Row_like_for_blocks known_tags other_tags mode).
    split; [ reflexivity | exact Hin ]. }
  assert (Hin_others :
    ∀ (index : Index.t)
      (known_tags : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
      (other_tags : Or_bottom.t row_like_block_case) (mode : Alloc_mode.t),
      In (index, Row_like_for_blocks known_tags other_tags mode) kept ->
      In (index, other_tags) others).
  { intros index known_tags other_tags mode Hin. unfold others.
    rewrite <- in_rev. apply in_map_iff.
    exists (index, Row_like_for_blocks known_tags other_tags mode).
    split; [ reflexivity | exact Hin ]. }
  assert (Hargs_knowns :
    args_in_scope
      (known_in_scope Z.eqb Or_unknown.Known (of_known block_case_variables)
         (of_known block_case_aliases))
      s knowns).
  { intros index m Hin. unfold knowns in Hin. rewrite <- in_rev in Hin.
    apply in_map_iff in Hin as ([index' blocks] & [= <- <-] & Hin).
    apply Hkept in Hin as [Hin _].
    destruct (Hargs index' blocks Hin) as (use & Huse & Hok).
    exists use. split; [ exact Huse | ].
    destruct (blocks_parts_in_scope use blocks Hok) as [Hknown Hother].
    split; [ exact Hknown | ]. intros o Ho. exact (Hother o Ho). }
  assert (Hargs_others :
    args_in_scope (arg_ok block_case_variables block_case_aliases) s
      (non_bottom [] others)).
  { intros index o Hin.
    apply (proj2 (non_bottom_spec others Hothers_distinct)) in Hin.
    unfold others in Hin. rewrite <- in_rev in Hin.
    apply in_map_iff in Hin as ([index' blocks] & [= <- Ho] & Hin).
    apply Hkept in Hin as [Hin _].
    destruct (Hargs index' blocks Hin) as (use & Huse & Hok).
    exists use. split; [ exact Huse | ].
    exact (proj2 (blocks_parts_in_scope use blocks Hok) o Ho). }
  pose proof
    (n_way_join_knowns_is_sound
       (n_way_join_int_indexed_product import_type type_eqb n_way_join_rec)
       Nat.eqb Nat.min (n_way_join_block_shape block_shape_eqb)
       variables_of_fields aliases_of_fields block_index_holds
       block_fields_hold block_index_sound block_index_bottom
       block_fields_sound)
    as Hknowns_sound.
  pose proof
    (n_way_join_case_is_sound
       (n_way_join_int_indexed_product import_type type_eqb n_way_join_rec)
       Nat.eqb Nat.min (n_way_join_block_shape block_shape_eqb)
       variables_of_fields aliases_of_fields block_index_holds
       block_fields_hold block_index_sound block_index_bottom
       block_fields_sound)
    as Hcase_sound.
  unfold map_join_result, n_way_join_row_like in Hstep.
  destruct
    (generic_merge_map_known Z.eqb Or_unknown.Known
       (n_way_join_knowns join_env_n_way_join_env_extension
          (n_way_join_int_indexed_product import_type type_eqb n_way_join_rec)
          Nat.eqb Nat.min (n_way_join_block_shape block_shape_eqb))
       s knowns)
    as [[known' sa] | ] eqn:Hmerge; [ | discriminate ].
  cbv beta iota in Hstep.
  destruct
    (generic_merge_map_known_is_sound Z.eqb z_eqb_decides_equality
       Or_unknown.Known (of_known block_case_variables)
       (of_known block_case_aliases)
       (or_unknown_bound (all_kept block_case_holds block_case_holds)) _
       Hknowns_sound s s sa knowns known' Hwf Hkinded (grows_refl s) Hvalid
       Hknowns_distinct Hargs_knowns Hmerge)
    as (Hvalida & Hgrowsa & Hka & Hin_known & Hfound & Hnone).
  assert (Hrow_shapes :
    with_checks = true ->
    ∀ (index : Index.t) (blocks : row_like_for_blocks),
      In (index, blocks) kept ->
      blocks_shape_ok (row_like_known_tags blocks)
        (row_like_other_tags blocks)
      = true).
  { intros Hchecks index [known_tags other_tags mode] Hin.
    apply Hkept in Hin as [Hin _].
    exact (arg_check_passes _ _ s row_likes index
             (Row_like_for_blocks known_tags other_tags mode) _ Hargs Hin
             (incl_appl _ (incl_refl _)) Hchecks). }
  assert (Hothers_shapes :
    with_checks = true ->
    ∀ (index : Index.t) (o : row_like_block_case),
      In (index, o) (non_bottom [] others) -> block_case_shape_ok o = true).
  { intros Hchecks index o Hin.
    apply (proj2 (non_bottom_spec others Hothers_distinct)) in Hin.
    unfold others in Hin. rewrite <- in_rev in Hin.
    apply in_map_iff in Hin as ([index' blocks] & [= <- Ho] & Hin).
    specialize (Hrow_shapes Hchecks index' blocks Hin).
    unfold blocks_shape_ok in Hrow_shapes.
    apply andb_true_iff in Hrow_shapes as [_ Hrow_shapes].
    rewrite Ho in Hrow_shapes. exact Hrow_shapes. }
  assert (Hknown_shape :
    with_checks = true ->
    Keyed.distinct Z.eqb (map fst known') = true
    ∧ ∀ (tag : Z) (v : Or_unknown.t row_like_block_case),
        Keyed.lookup Z.eqb tag known' = Some v ->
        match v with
        | Or_unknown.Known c => block_case_shape_ok c = true
        | Or_unknown.Unknown => True
        end).
  { intros Hchecks.
    refine
      (generic_merge_map_known_shape Z.eqb z_eqb_decides_equality
         Or_unknown.Known _
         (fun (_ : Z) (v : Or_unknown.t row_like_block_case) =>
            match v with
            | Or_unknown.Known c => block_case_shape_ok c = true
            | Or_unknown.Unknown => True
            end)
         s sa knowns known' Hknowns_distinct _ _ Hmerge).
    - intros key s1 s2 vals v Hjoin Hvals.
      exact (block_knowns_join_shape s1 s2 vals v Hvals Hjoin).
    - intros index m Hin. unfold knowns in Hin. rewrite <- in_rev in Hin.
      apply in_map_iff in Hin as ([index' blocks] & [= <- <-] & Hin).
      specialize (Hrow_shapes Hchecks index' blocks Hin).
      unfold blocks_shape_ok in Hrow_shapes.
      apply andb_true_iff in Hrow_shapes as [Hrow_shapes Hother_ok].
      apply andb_true_iff in Hrow_shapes as [_ Hcases_ok].
      cbn [fst snd]. split.
      + intros key v Hlookup.
        apply (keyed_for_all_bindings_b Z Z.eqb z_eqb_decides_equality)
          in Hcases_ok.
        assert (Hv :=
          proj1 (keyed_for_all_bindings_lookup Z Z.eqb z_eqb_decides_equality
                   _ _ _)
            Hcases_ok key v Hlookup).
        destruct v; [ exact Hv | exact I ].
      + intros o key Ho. rewrite Ho in Hother_ok. exact Hother_ok. }
  assert (Hfinal :
    ∀ (other' : Or_bottom.t row_like_block_case) (sb : state),
      state_valid sb -> grows sa sb ->
      (state_kinded sa -> state_kinded sb) ->
      (with_checks = true ->
       match other' with
       | Or_bottom.Ok c => block_case_shape_ok c = true
       | Or_bottom.Bottom => True
       end) ->
      match other' with
      | Or_bottom.Ok c =>
        answer_holds_from (of_known block_case_variables)
          (of_known block_case_aliases)
          (known_bound (all_kept block_case_holds block_case_holds)) sa sb
          (non_bottom [] others) (Or_unknown.Known c)
      | Or_bottom.Bottom => non_bottom [] others = []
      end ->
      state_valid sb ∧ grows s sb
      ∧ in_target (of_known free_variables_of_row_like_for_blocks) sb
          (Or_unknown.Known (Row_like_for_blocks known' other' alloc_mode))
      ∧ (∀ (index : Index.t) (x : row_like_for_blocks) (use : Typing_env.t)
           (statics : static_env) (valuation valuation' : env),
           In (index, x) row_likes ->
           joined_env sb index = Some use ->
           typing_env_satisfied_strictly statics valuation use ->
           extends sb index valuation valuation' ->
           known_bound (all_kept blocks_hold blocks_hold) statics valuation x
             valuation'
             (Or_unknown.Known (Row_like_for_blocks known' other' alloc_mode)))
      ∧ (state_kinded s ->
         state_kinded sb
         ∧ kinded_in_target (of_known kinds_of_blocks) sb
             (Or_unknown.Known
                (Row_like_for_blocks known' other' alloc_mode)))).
  { intros other' sb Hvalidb Hgrowsb Hkb Hother_shape Hother'.
    split; [ exact Hvalidb | ].
    split; [ exact (grows_trans _ _ _ Hgrowsa Hgrowsb) | ].
    split; [ | split ].
    - intros x Hx. cbn [of_known] in Hx. simpl in Hx.
      apply in_app_or in Hx as [Hx | Hx].
      + apply in_flat_map in Hx as ([key y] & Hin & Hx). cbn [snd] in Hx.
        destruct (Hfound key y (Hin_known key y Hin)) as (vals & _ & Hholds).
        destruct (Hholds sb Hgrowsb) as (Htarget & _ & _). apply Htarget.
        destruct y; exact Hx.
      + destruct other' as [c | ]; [ | destruct Hx ].
        destruct (Hother' sb (grows_refl sb)) as (Htarget & _ & _).
        exact (Htarget x Hx).
    - intros index [known_tags other_tags mode] use statics valuation
        valuation' Hin Huse Hsatisfied Hextends w Hw.
      unfold blocks_hold in *.
      assert (Hin_kept :
        In (index, Row_like_for_blocks known_tags other_tags mode) kept).
      { apply Hkept. split; [ exact Hin | ].
        destruct (row_like_for_blocks_is_bottom _) eqn:Hb; [ | reflexivity ].
        destruct
          (row_like_for_blocks_is_bottom_satisfied statics valuation (fst w)
             (snd w) _ Hb Hw). }
      apply row_like_for_blocks_find_iff in Hw.
      apply row_like_for_blocks_find_iff. unfold row_like_block_case in *.
      destruct (Keyed.lookup Z.eqb (fst w) known') as [y | ] eqn:Hy.
      + destruct (Hfound _ y Hy) as (vals & Hvals & Hholds).
        destruct (Hholds sb Hgrowsb) as (_ & Hbound & _).
        destruct
          (gather_known_spec Z.eqb z_eqb_decides_equality Or_unknown.Known
             knowns (fst w) vals Hknowns_distinct Hvals)
          as (_ & _ & Hto & _).
        destruct
          (Hto (index, (known_tags, other_tags))
             (Hin_knowns _ _ _ _ Hin_kept))
          as (Hsome & Hnone'). cbn [fst snd] in Hsome, Hnone'.
        destruct y as [c' | ]; [ | exact I ].
        destruct (Keyed.lookup Z.eqb (fst w) known_tags) as [v | ] eqn:Hv.
        * specialize
            (Hbound index v use statics valuation valuation' (Hsome v eq_refl)
               Huse Hsatisfied Hextends).
          destruct v as [c | ]; [ exact (Hbound w Hw) | destruct Hbound ].
        * destruct other_tags as [o | ]; [ | destruct Hw ].
          exact
            (Hbound index (Or_unknown.Known o) use statics valuation
               valuation' (Hnone' eq_refl o eq_refl) Huse Hsatisfied Hextends w
               Hw).
      + pose proof
          (gather_known_none Z.eqb z_eqb_decides_equality Or_unknown.Known
             knowns (fst w) (Hnone _ Hy) (index, (known_tags, other_tags))
             (Hin_knowns _ _ _ _ Hin_kept))
          as Hv.
        cbn [fst snd] in Hv. rewrite Hv in Hw.
        destruct other_tags as [o | ]; [ | destruct Hw ].
        assert (Hin_other : In (index, o) (non_bottom [] others)).
        { apply (proj2 (non_bottom_spec others Hothers_distinct)).
          exact (Hin_others _ _ _ _ Hin_kept). }
        destruct other' as [c | ].
        * destruct (Hother' sb (grows_refl sb)) as (_ & Hbound & _).
          exact
            (Hbound index o use statics valuation valuation' Hin_other Huse
               Hsatisfied Hextends w Hw).
        * rewrite Hother' in Hin_other. destruct Hin_other.
    - intros Hk. split; [ exact (Hkb (Hka Hk)) | ].
      intros simple kind Hsimple. cbn [of_known] in Hsimple.
      simpl in Hsimple. apply in_app_or in Hsimple as [Hsimple | Hsimple].
      { refine (answer_check_passes sb _ simple kind _ Hsimple).
        intros Hchecks. destruct (Hknown_shape Hchecks) as [Hdistinct' Hcases'].
        unfold blocks_shape_ok. rewrite !andb_true_iff.
        split; [ split; [ exact Hdistinct' | ] | ].
        - apply (keyed_for_all_bindings_b Z Z.eqb z_eqb_decides_equality).
          apply (keyed_for_all_bindings_lookup Z Z.eqb z_eqb_decides_equality).
          intros tag v Hlookup. specialize (Hcases' tag v Hlookup).
          destruct v; [ exact Hcases' | reflexivity ].
        - specialize (Hother_shape Hchecks).
          destruct other'; [ exact Hother_shape | reflexivity ]. }
      apply in_app_or in Hsimple as [Hsimple | Hsimple].
      + apply in_flat_map in Hsimple as ([key y] & Hin & Hsimple).
        cbn [snd] in Hsimple.
        destruct (Hfound key y (Hin_known key y Hin)) as (vals & _ & Hholds).
        destruct (Hholds sb Hgrowsb) as (_ & _ & Hkinds). apply (Hkinds Hk).
        destruct y; exact Hsimple.
      + destruct other' as [c | ]; [ | destruct Hsimple ].
        destruct (Hother' sb (grows_refl sb)) as (_ & _ & Hkinds).
        exact (Hkinds (Hka Hk) simple kind Hsimple). }
  match type of Hstep with
  | context [n_way_join_case ?a ?b ?c ?d ?e sa ?l] =>
    destruct (n_way_join_case a b c d e sa l) as [result | ] eqn:Hother;
      [ | discriminate ]
  end.
  pose proof
    (Hcase_sound s sa (non_bottom [] others) result Hwf Hkinded Hgrowsa
       Hvalida (proj1 (non_bottom_spec others Hothers_distinct)) Hargs_others
       Hother)
    as Hresult.
  destruct result as [[[c | ] sb] | ]; cbv beta iota in Hstep;
    unfold or_unknown_map in Hstep; cbn [fst snd] in Hstep.
  all: injection Hstep as <- <-.
  - destruct Hresult as (Hvalidb & Hgrowsb & Hkb & Hholds).
    exact
      (Hfinal (Or_bottom.Ok c) sb Hvalidb Hgrowsb Hkb
         (fun Hchecks =>
            block_case_join_shape sa sb (non_bottom [] others) c
              (Hothers_shapes Hchecks) Hother)
         Hholds).
  - destruct Hresult as (Hvalidb & Hgrowsb & Hkb & _).
    split; [ exact Hvalidb | ].
    split; [ exact (grows_trans _ _ _ Hgrowsa Hgrowsb) | ].
    split; [ intros x [] | ]. split; [ intros; exact I | ].
    intros Hk. split; [ exact (Hkb (Hka Hk)) | intros simple kind [] ].
  - exact
      (Hfinal Or_bottom.Bottom sa Hvalida (grows_refl sa) (fun H => H)
         (fun _ => I) Hresult).
Qed.

(* ------------------------------------------------------------------------ *)
(* Row-like structures of closures                                          *)
(* ------------------------------------------------------------------------ *)

(* ------------------------------------------------------------------------ *)
(* The shape of the cases of closures that are answered                     *)
(* ------------------------------------------------------------------------ *)

(* What is gathered binds a slot when every product does. *)
Lemma gather_components_bound {A : Type} (g : positive)
    (products : list (Index.t * finite_map A)) :
  products <> [] ->
  is_bound g (gather_components products) = true
  <-> ∀ (id : Index.t) (product : finite_map A),
        In (id, product) products -> is_bound g product = true.
Proof.
  destruct products as [ | [id1 p1] rest]; intros Hne;
    [ destruct (Hne eq_refl) | clear Hne ].
  unfold is_bound, gather_components.
  rewrite gather_components_fold, lookup_map_values.
  destruct (lookup g p1) as [v | ] eqn:Hv; cbn [option_map].
  - assert (Hrest :
      (∃ r, indexed_map_opt (fun _ product => lookup g product) rest = Some r)
      <-> ∀ (id : Index.t) (product : finite_map A),
            In (id, product) rest ->
            match lookup g product with
            | Some _ => true
            | None => false
            end
            = true).
    { clear Hv. induction rest as [ | [id2 p2] rest IH]; simpl.
      - split; [ intros _ id product [] | intros _; exists []; reflexivity ].
      - destruct (lookup g p2) as [v2 | ] eqn:Hv2.
        + destruct (indexed_map_opt (fun _ product => lookup g product) rest)
            as [r | ].
          * split; [ | intros _; eexists; reflexivity ].
            intros _ id product [[= <- <-] | Hin];
              [ rewrite Hv2; reflexivity | ].
            exact (proj1 IH (ex_intro _ r eq_refl) id product Hin).
          * split; [ intros [r' [=]] | ]. intros Hall.
            destruct (proj2 IH (fun id product Hin =>
                                  Hall id product (or_intror Hin)))
              as [r' [=]].
        + split; [ intros [r' [=]] | ]. intros Hall.
          specialize (Hall id2 p2 (or_introl eq_refl)). rewrite Hv2 in Hall.
          discriminate Hall. }
    destruct (indexed_map_opt (fun _ product => lookup g product) rest)
      as [r | ].
    + split; [ | reflexivity ].
      intros _ id product [[= <- <-] | Hin]; [ rewrite Hv; reflexivity | ].
      exact (proj1 Hrest (ex_intro _ r eq_refl) id product Hin).
    + split; [ discriminate | ]. intros Hall.
      destruct (proj2 Hrest (fun id product Hin =>
                               Hall id product (or_intror Hin)))
        as [r' [=]].
  - split; [ discriminate | ]. intros Hall.
    specialize (Hall id1 p1 (or_introl eq_refl)). rewrite Hv in Hall.
    discriminate Hall.
Qed.

(* It binds no slot twice, if no product does. *)
Lemma gather_components_keys {A : Type}
    (products : list (Index.t * finite_map A)) :
  (∀ (id : Index.t) (product : finite_map A),
     In (id, product) products -> distinct (keys product) = true) ->
  distinct (keys (gather_components products)) = true.
Proof.
  destruct products as [ | [id1 p1] rest]; intros Hdistinct; [ reflexivity | ].
  unfold gather_components.
  assert (Hfirst :
    distinct (keys (map_values (fun val1 : A => [(id1, val1)]) p1)) = true).
  { unfold keys, map_values. rewrite map_map.
    exact (Hdistinct id1 p1 (or_introl eq_refl)). }
  clear Hdistinct. revert Hfirst.
  generalize (map_values (fun val1 : A => [(id1, val1)]) p1).
  induction rest as [ | [id2 p2] rest IH]; intros acc Hacc; simpl;
    [ exact Hacc | ].
  apply IH. unfold merge, keys. apply tabulate_distinct. apply dedup_distinct.
Qed.

(* The product answered binds no slot twice, and the slots that it
   binds, every product given binds. *)
Lemma generic_product_keys (s s' : state)
    (products : list (Index.t * finite_map flambda_type))
    (out : finite_map flambda_type) :
  n_way_join_generic_product n_way_join_rec s products = Some (out, s') ->
  distinct (keys out) = true
  ∧ ∀ g : positive,
      is_bound g out = true ->
      ∀ (id : Index.t) (product : finite_map flambda_type),
        In (id, product) products -> is_bound g product = true.
Proof.
  unfold n_way_join_generic_product.
  destruct products as [ | p0 rest] eqn:Hproducts.
  - intros [= <- _]. split; [ reflexivity | intros g [=] ].
  - rewrite <- Hproducts. intros Hfold.
    unfold Keyed.filter_map_fold, Keyed.filter_map_fold_on in Hfold.
    destruct
      (tabulate_fold_opt_keys Pos.eqb _ _ s out s'
         (keyed_domain_distinct positive Pos.eqb pos_eqb_decides_equality _ _)
         Hfold)
      as [Hdistinct Hsub].
    split; [ exact Hdistinct | ].
    intros g Hg. rewrite is_bound_among in Hg. specialize (Hsub g Hg).
    rewrite (keyed_among_domain positive Pos.eqb pos_eqb_decides_equality)
      in Hsub.
    apply (gather_components_bound g products); [ | exact Hsub ].
    rewrite Hproducts. discriminate.
Qed.

(* The contents that an intersection of indexes has. *)
Lemma closures_inter_among (g : positive)
    (rest : list (row_like_index set_of_closures_contents unit)) :
  ∀ contents : set_of_closures_contents,
    let inter :=
      List.fold_left
        (fun i1' i2 =>
           match index_domain i2 with
           | Row_like_index_domain.At_least i2'
           | Row_like_index_domain.Known i2' =>
             set_of_closures_contents_inter i1' i2'
           end)
        rest contents
    in
    among g (contents_closures inter)
    = among g (contents_closures contents)
      && forallb
           (fun i => among g (contents_closures (closures_index_contents i)))
           rest
    ∧ among g (contents_value_slots inter)
      = among g (contents_value_slots contents)
        && forallb
             (fun i =>
                among g (contents_value_slots (closures_index_contents i)))
             rest.
Proof.
  induction rest as [ | i rest IH]; intros contents; cbv zeta; simpl.
  - rewrite !andb_true_r. split; reflexivity.
  - destruct (IH (match index_domain i with
                  | Row_like_index_domain.At_least i2'
                  | Row_like_index_domain.Known i2' =>
                    set_of_closures_contents_inter contents i2'
                  end)) as [Hc Hv].
    cbv zeta in Hc, Hv. rewrite Hc, Hv. unfold closures_index_contents.
    destruct (index_domain i) as [c | c]; simpl;
      rewrite !(keyed_among_inter positive Pos.eqb pos_eqb_decides_equality),
        !andb_assoc;
      split; reflexivity.
Qed.

(* The index answered for indexes of closures: its contents are those
   that every index has, and if it is exact so is every index, with
   the same closures. *)
Lemma closures_index_join_shape
    (i1 : row_like_index set_of_closures_contents unit)
    (rest : list (row_like_index set_of_closures_contents unit))
    (index : row_like_index set_of_closures_contents unit) :
  n_way_join_index set_of_closures_contents_equal
    set_of_closures_contents_inter (fun _ => Ok tt) (i1 :: rest)
  = Ok index ->
  (∀ g : positive,
     (∀ i : row_like_index set_of_closures_contents unit,
        In i (i1 :: rest) ->
        among g (contents_closures (closures_index_contents i)) = true) ->
     among g (contents_closures (closures_index_contents index)) = true)
  ∧ (∀ g : positive,
       (∀ i : row_like_index set_of_closures_contents unit,
          In i (i1 :: rest) ->
          among g (contents_value_slots (closures_index_contents i)) = true) ->
       among g (contents_value_slots (closures_index_contents index)) = true)
  ∧ ∀ contents : set_of_closures_contents,
      index_domain index = Row_like_index_domain.Known contents ->
      ∀ i : row_like_index set_of_closures_contents unit,
        In i (i1 :: rest) ->
        ∃ contents' : set_of_closures_contents,
          index_domain i = Row_like_index_domain.Known contents'
          ∧ same_names (contents_closures contents)
              (contents_closures contents').
Proof.
  unfold n_way_join_index. cbv beta.
  assert (Hat_least :
    ∀ c1 : set_of_closures_contents,
      closures_index_contents i1 = c1 ->
      ∀ contents : set_of_closures_contents,
        contents
        = List.fold_left
            (fun i1' i2 =>
               match index_domain i2 with
               | Row_like_index_domain.At_least i2'
               | Row_like_index_domain.Known i2' =>
                 set_of_closures_contents_inter i1' i2'
               end)
            rest c1 ->
        (∀ g : positive,
           (∀ i : row_like_index set_of_closures_contents unit,
              In i (i1 :: rest) ->
              among g (contents_closures (closures_index_contents i)) = true) ->
           among g (contents_closures contents) = true)
        ∧ (∀ g : positive,
             (∀ i : row_like_index set_of_closures_contents unit,
                In i (i1 :: rest) ->
                among g (contents_value_slots (closures_index_contents i))
                = true) ->
             among g (contents_value_slots contents) = true)).
  { intros c1 Hc1 contents ->.
    split; intros g Hall;
      [ rewrite (proj1 (closures_inter_among g rest c1))
      | rewrite (proj2 (closures_inter_among g rest c1)) ];
      apply andb_true_iff;
      (split;
       [ rewrite <- Hc1; exact (Hall i1 (or_introl eq_refl))
       | apply forallb_forall; intros i Hi; exact (Hall i (or_intror Hi)) ]). }
  destruct (index_domain i1) as [c1 | c1] eqn:H1.
  - destruct (forallb _ rest) eqn:Hall; intros [= <-]; cbn [index_domain].
    + unfold closures_index_contents at 2 4. cbn [index_domain].
      split; [ | split ].
      * intros g Hg. specialize (Hg i1 (or_introl eq_refl)).
        unfold closures_index_contents in Hg. rewrite H1 in Hg. exact Hg.
      * intros g Hg. specialize (Hg i1 (or_introl eq_refl)).
        unfold closures_index_contents in Hg. rewrite H1 in Hg. exact Hg.
      * intros contents [= <-] i [<- | Hi].
        -- exists c1. split; [ exact H1 | intros name; reflexivity ].
        -- rewrite forallb_forall in Hall. specialize (Hall i Hi).
           destruct (index_domain i) as [c | c]; [ | discriminate Hall ].
           exists c. split; [ reflexivity | ].
           unfold set_of_closures_contents_equal in Hall.
           apply andb_true_iff in Hall as [Hall _].
           exact (proj1 (keyed_same_keys_b positive Pos.eqb
                           pos_eqb_decides_equality _ _) Hall).
    + destruct (Hat_least c1) with (contents := List.fold_left
            (fun i1' i2 =>
               match index_domain i2 with
               | Row_like_index_domain.At_least i2'
               | Row_like_index_domain.Known i2' =>
                 set_of_closures_contents_inter i1' i2'
               end)
            rest c1) as [Hc Hv];
        [ unfold closures_index_contents; rewrite H1; reflexivity
        | reflexivity | ].
      split; [ exact Hc | ]. split; [ exact Hv | intros contents [=] ].
  - intros [= <-]. cbn [index_domain].
    destruct (Hat_least c1) with (contents := List.fold_left
            (fun i1' i2 =>
               match index_domain i2 with
               | Row_like_index_domain.At_least i2'
               | Row_like_index_domain.Known i2' =>
                 set_of_closures_contents_inter i1' i2'
               end)
            rest c1) as [Hc Hv];
      [ unfold closures_index_contents; rewrite H1; reflexivity
      | reflexivity | ].
    split; [ exact Hc | ]. split; [ exact Hv | intros contents [=] ].
Qed.

(* The entry answered for entries: no map of it binds a slot twice,
   its closure types are of kind value, and the slots that its maps
   bind are bound by the same map of every entry. For the function
   types, those are all of them. *)
Lemma closures_entry_join_slots (s s' : state)
    (entries : list (Index.t * closures_entry)) (entry : closures_entry) :
  entries <> [] ->
  n_way_join_closures_entry n_way_join_rec s entries = Some (entry, s') ->
  (∀ (id : Index.t) (e : closures_entry),
     In (id, e) entries ->
     distinct (keys (closures_entry_function_types e)) = true) ->
  distinct (keys (closures_entry_function_types entry)) = true
  ∧ distinct (keys (closures_entry_closure_types entry)) = true
  ∧ distinct (keys (closures_entry_value_slot_types entry)) = true
  ∧ for_all_bindings_b (fun _ ty => kind_eqb (kind ty) Value)
      (closures_entry_closure_types entry)
    = true
  ∧ (∀ g : function_slot,
       is_bound g (closures_entry_function_types entry) = true
       <-> ∀ (id : Index.t) (e : closures_entry),
             In (id, e) entries ->
             is_bound g (closures_entry_function_types e) = true)
  ∧ (∀ g : function_slot,
       is_bound g (closures_entry_closure_types entry) = true ->
       ∀ (id : Index.t) (e : closures_entry),
         In (id, e) entries ->
         is_bound g (closures_entry_closure_types e) = true)
  ∧ ∀ w : value_slot,
      is_bound w (closures_entry_value_slot_types entry) = true ->
      ∀ (id : Index.t) (e : closures_entry),
        In (id, e) entries ->
        is_bound w (closures_entry_value_slot_types e) = true.
Proof.
  intros Hne Hjoin Hdistinct. unfold n_way_join_closures_entry in Hjoin.
  cbv zeta in Hjoin.
  destruct (n_way_join_function_slot_indexed_product _ _ _)
    as [[closure_types s1] | ] eqn:Hclosures; [ | discriminate ].
  cbn in Hjoin.
  destruct (n_way_join_value_slot_indexed_product _ _ _)
    as [[value_slot_types s2] | ] eqn:Hvalue_slots; [ | discriminate ].
  cbn in Hjoin. injection Hjoin as <- _.
  cbn [closures_entry_function_types closures_entry_closure_types
         closures_entry_value_slot_types].
  unfold n_way_join_function_slot_indexed_product in Hclosures.
  destruct (n_way_join_generic_product _ _ _) as [[components s3] | ]
    eqn:Hcomponents; [ | discriminate ].
  cbn in Hclosures.
  destruct (for_all_bindings_b _ components) eqn:Hkinds; [ | discriminate ].
  injection Hclosures as <- _.
  unfold n_way_join_value_slot_indexed_product in Hvalue_slots.
  destruct (generic_product_keys _ _ _ _ Hcomponents) as [Hdc Hbc].
  destruct (generic_product_keys _ _ _ _ Hvalue_slots) as [Hdv Hbv].
  split.
  { unfold n_way_join_function_types, keys, map_values. rewrite map_map.
    apply (gather_components_keys (A := Or_unknown.t function_type)).
    intros id m Hin. apply in_map_iff in Hin as ([id' e] & [= <- <-] & Hin).
    exact (Hdistinct id' e Hin). }
  split; [ exact Hdc | ]. split; [ exact Hdv | ]. split; [ exact Hkinds | ].
  split; [ | split ].
  - intros g.
    assert (Hbound :
      is_bound g
        (n_way_join_function_types
           (map
              (fun entry =>
                 (fst entry, closures_entry_function_types (snd entry)))
              entries))
      = is_bound g
          (gather_components
             (map
                (fun entry =>
                   (fst entry, closures_entry_function_types (snd entry)))
                entries))).
    { unfold n_way_join_function_types, is_bound.
      rewrite lookup_map_values.
      change (gather_function_types ?l) with (gather_components l).
      destruct (lookup g _); reflexivity. }
    rewrite Hbound, gather_components_bound;
      [ | destruct entries; [ destruct (Hne eq_refl) | discriminate ] ].
    split.
    + intros Hall id e Hin.
      exact
        (Hall id (closures_entry_function_types e)
           (in_map
              (fun entry =>
                 (fst entry, closures_entry_function_types (snd entry)))
              entries (id, e) Hin)).
    + intros Hall id m Hin.
      apply in_map_iff in Hin as ([id' e] & [= <- <-] & Hin).
      exact (Hall id' e Hin).
  - intros g Hg id e Hin. apply (Hbc g Hg id). rewrite <- in_rev.
    exact
      (in_map
         (fun entry => (fst entry, closures_entry_closure_types (snd entry)))
         entries (id, e) Hin).
  - intros w Hw id e Hin. apply (Hbv w Hw id). rewrite <- in_rev.
    exact
      (in_map
         (fun entry =>
            (fst entry, closures_entry_value_slot_types (snd entry)))
         entries (id, e) Hin).
Qed.

(* A case of closures under a function slot: conditions 5, 8 and 9. *)
Definition closures_case_shape_ok (slot : function_slot)
    (case : row_like_closures_case) : bool :=
  closures_index_has_slot slot (case_index case)
  && closures_entry_shape_ok (case_index case) (case_maps_to case).

(* The case answered for such cases is such a case. *)
Lemma closures_case_join_shape (slot : function_slot) (s s' : state)
    (cases : list (Index.t * row_like_closures_case))
    (c : row_like_closures_case) :
  (∀ (id : Index.t) (c0 : row_like_closures_case),
     In (id, c0) cases -> closures_case_shape_ok slot c0 = true) ->
  n_way_join_case join_env_n_way_join_env_extension
    (fun env (_ : unit) entries =>
       n_way_join_closures_entry n_way_join_rec env entries)
    set_of_closures_contents_equal set_of_closures_contents_inter
    (fun _ => Ok tt) s cases
  = Some (Or_bottom.Ok (Or_unknown.Known c, s')) ->
  closures_case_shape_ok slot c = true.
Proof.
  intros Hok Hjoin. unfold n_way_join_case in Hjoin.
  destruct cases as [ | [id1 c1] rest]; [ discriminate Hjoin | ].
  set (cases := (id1, c1) :: rest) in *.
  destruct
    (n_way_join_index set_of_closures_contents_equal
       set_of_closures_contents_inter (fun _ => Ok tt)
       (map (fun case => case_index (snd case)) cases))
    as [ | index | ] eqn:Hindex; try discriminate.
  destruct
    (n_way_join_closures_entry n_way_join_rec s
       (map (fun case => (fst case, case_maps_to (snd case))) cases))
    as [[entry s1] | ] eqn:Hentry; [ | discriminate ].
  cbn in Hjoin.
  destruct (n_way_join_env_extension join_env_n_way_join_env_extension s1 _)
    as [[extension s2] | ]; [ | discriminate ].
  cbn in Hjoin. injection Hjoin as <- _.
  (* What the cases given have. *)
  assert (Hcases :
    ∀ (id : Index.t) (c0 : row_like_closures_case),
      In (id, c0) cases ->
      closures_index_has_slot slot (case_index c0) = true
      ∧ closures_entry_slots_fit (case_index c0)
          (domain (closures_entry_function_types (case_maps_to c0)))
          (domain (closures_entry_closure_types (case_maps_to c0)))
          (domain (closures_entry_value_slot_types (case_maps_to c0)))
        = true
      ∧ distinct (keys (closures_entry_function_types (case_maps_to c0)))
        = true).
  { intros id c0 Hin. specialize (Hok id c0 Hin).
    unfold closures_case_shape_ok in Hok.
    apply andb_true_iff in Hok as [Hslot Hentry0]. split; [ exact Hslot | ].
    unfold closures_entry_shape_ok in Hentry0.
    destruct (case_maps_to c0) as [ft ct vt].
    rewrite !andb_true_iff in Hentry0.
    destruct Hentry0 as [[[[Hfit Hd] _] _] _].
    split; [ exact Hfit | exact Hd ]. }
  assert (Hin_indexes :
    ∀ i : row_like_index set_of_closures_contents unit,
      In i (map (fun case => case_index (snd case)) cases) ->
      ∃ (id : Index.t) (c0 : row_like_closures_case),
        In (id, c0) cases ∧ i = case_index c0).
  { intros i Hi. apply in_map_iff in Hi as ([id c0] & <- & Hin).
    exists id, c0. split; [ exact Hin | reflexivity ]. }
  assert (Hin_entries :
    ∀ (id : Index.t) (c0 : row_like_closures_case),
      In (id, c0) cases ->
      In (id, case_maps_to c0)
        (map (fun case => (fst case, case_maps_to (snd case))) cases)).
  { intros id c0 Hin.
    exact
      (in_map (fun case => (fst case, case_maps_to (snd case))) cases
         (id, c0) Hin). }
  destruct (closures_index_join_shape _ _ _ Hindex)
    as (Hclosures & Hvalue_slots & Hexact).
  assert (Hne :
    map (fun case => (fst case, case_maps_to (snd case))) cases <> []).
  { unfold cases. discriminate. }
  destruct (closures_entry_join_slots s s1 _ entry Hne Hentry)
    as (Hdf & Hdc & Hdv & Hkinds & Hbf & Hbc & Hbv).
  { intros id e Hin. apply in_map_iff in Hin as ([id' c0] & [= <- <-] & Hin).
    exact (proj2 (proj2 (Hcases id' c0 Hin))). }
  unfold closures_case_shape_ok. cbn [case_index case_maps_to].
  apply andb_true_iff. split.
  { unfold closures_index_has_slot.
    destruct (index_domain index) as [contents | contents] eqn:Hdomain;
      [ | reflexivity ].
    destruct (Hexact contents eq_refl (case_index c1) (or_introl eq_refl))
      as (contents1 & H1 & Hsame).
    rewrite (Hsame slot).
    assert (Hslot := proj1 (Hcases id1 c1 (or_introl eq_refl))).
    unfold closures_index_has_slot in Hslot. rewrite H1 in Hslot.
    exact Hslot. }
  unfold closures_entry_shape_ok. destruct entry as [ft ct vt].
  cbn [closures_entry_function_types closures_entry_closure_types
         closures_entry_value_slot_types] in *.
  rewrite Hdf, Hdc, Hdv, Hkinds, !andb_true_r.
  unfold closures_entry_slots_fit. rewrite !andb_true_iff.
  split; [ split; [ split | ] | ].
  - apply all_among. intros g Hg. rewrite among_domain in Hg.
    apply Hclosures. intros i Hi.
    destruct (Hin_indexes i Hi) as (id & c0 & Hin & ->).
    destruct (Hcases id c0 Hin) as (_ & Hfit & _).
    unfold closures_entry_slots_fit in Hfit. rewrite !andb_true_iff in Hfit.
    destruct Hfit as [[[Hfit _] _] _].
    apply (proj1 (all_among _ _) Hfit g). rewrite among_domain.
    exact (proj1 (Hbf g) Hg id (case_maps_to c0) (Hin_entries id c0 Hin)).
  - apply all_among. intros g Hg. rewrite among_domain in Hg.
    apply Hclosures. intros i Hi.
    destruct (Hin_indexes i Hi) as (id & c0 & Hin & ->).
    destruct (Hcases id c0 Hin) as (_ & Hfit & _).
    unfold closures_entry_slots_fit in Hfit. rewrite !andb_true_iff in Hfit.
    destruct Hfit as [[[_ Hfit] _] _].
    apply (proj1 (all_among _ _) Hfit g). rewrite among_domain.
    exact (Hbc g Hg id (case_maps_to c0) (Hin_entries id c0 Hin)).
  - apply all_among. intros w Hw. rewrite among_domain in Hw.
    apply Hvalue_slots. intros i Hi.
    destruct (Hin_indexes i Hi) as (id & c0 & Hin & ->).
    destruct (Hcases id c0 Hin) as (_ & Hfit & _).
    unfold closures_entry_slots_fit in Hfit. rewrite !andb_true_iff in Hfit.
    destruct Hfit as [[_ Hfit] _].
    apply (proj1 (all_among _ _) Hfit w). rewrite among_domain.
    exact (Hbv w Hw id (case_maps_to c0) (Hin_entries id c0 Hin)).
  - destruct (index_domain index) as [contents | contents] eqn:Hdomain;
      [ | reflexivity ].
    apply all_among. intros g Hg. rewrite among_domain.
    apply (proj2 (Hbf g)). intros id e Hin.
    apply in_map_iff in Hin as ([id' c0] & [= <- <-] & Hin).
    destruct
      (Hexact contents eq_refl (case_index c0)
         (in_map (fun case => case_index (snd case)) cases (id', c0) Hin))
      as (contents0 & H0 & Hsame).
    destruct (Hcases id' c0 Hin) as (_ & Hfit & _).
    unfold closures_entry_slots_fit in Hfit. rewrite !andb_true_iff in Hfit.
    destruct Hfit as [_ Hfit]. rewrite H0 in Hfit.
    rewrite <- among_domain. apply (proj1 (all_among _ _) Hfit g).
    rewrite <- (Hsame g). exact Hg.
Qed.

Lemma closures_knowns_join_shape (slot : function_slot) (s s' : state)
    (cases : list (Index.t * row_like_closures_case))
    (c : row_like_closures_case) :
  (∀ (id : Index.t) (c0 : row_like_closures_case),
     In (id, c0) cases -> closures_case_shape_ok slot c0 = true) ->
  n_way_join_knowns join_env_n_way_join_env_extension
    (fun env (_ : unit) entries =>
       n_way_join_closures_entry n_way_join_rec env entries)
    set_of_closures_contents_equal set_of_closures_contents_inter
    (fun _ => Ok tt) s
    (map (fun case => (fst case, Or_unknown.Known (snd case))) cases)
  = Some (Or_bottom.Ok (Or_unknown.Known c, s')) ->
  closures_case_shape_ok slot c = true.
Proof.
  intros Hok Hjoin. unfold n_way_join_knowns in Hjoin.
  rewrite all_known_eq in Hjoin.
  match type of Hjoin with
  | context [indexed_map_opt ?f ?l] =>
    destruct (indexed_map_opt f l) as [known | ] eqn:Hknown
  end.
  - apply (closures_case_join_shape slot s s' known c); [ | exact Hjoin ].
    intros id c0 Hin.
    destruct (indexed_map_opt_spec _ _ known Hknown) as (_ & _ & Hfrom).
    destruct (Hfrom id c0 Hin) as ([c' | ] & Heq & Hin');
      [ injection Heq as <- | discriminate ].
    apply in_map_iff in Hin' as ([id' c0'] & [= <- <-] & Hin').
    exact (Hok id' c0' Hin').
  - discriminate Hjoin.
Qed.

Definition closures_index_holds (w : closure_and_set)
    (index : row_like_index set_of_closures_contents unit) : Prop :=
  closures_index_satisfied (snd (fst w)) (snd w) index.

Definition closures_hold (statics : static_env) (vars : env)
    (w : closure_and_set) (closures : row_like_for_closures) : Prop :=
  let '(v, slot, functions, value_slots) := w in
  row_like_for_closures_satisfied statics vars v slot functions value_slots
    closures.

Local Abbreviation closures_case_variables :=
  (variables_of_case (lattice := set_of_closures_contents) (shape := unit)
     free_variables_of_closures_entry).
Local Abbreviation closures_case_aliases :=
  (aliases_of_case (lattice := set_of_closures_contents) (shape := unit)
     kinds_of_entry).
Local Abbreviation closures_case_holds :=
  (case_holds closures_index_holds closures_entry_holds).

Lemma closures_index_sound
    (indexes : list (row_like_index set_of_closures_contents unit))
    (index : row_like_index set_of_closures_contents unit) :
  n_way_join_index set_of_closures_contents_equal
    set_of_closures_contents_inter (fun _ => Ok tt) indexes
  = Ok index ->
  ∀ i : row_like_index set_of_closures_contents unit,
    In i indexes ->
    ∀ w : closure_and_set,
      closures_index_holds w i -> closures_index_holds w index.
Proof.
  intros Hjoined i Hin w Hdomain.
  refine
    (proj1
       (n_way_join_index_sound set_of_closures_contents_equal
          set_of_closures_contents_inter (fun _ => Ok tt)
          (fun (w : closure_and_set) contents =>
             same_names (contents_closures contents) (domain (snd (fst w)))
             ∧ same_names (contents_value_slots contents) (domain (snd w)))
          (fun (w : closure_and_set) contents =>
             FlambdaMapSpec.all_among (contents_closures contents)
               (domain (snd (fst w)))
             = true
             ∧ FlambdaMapSpec.all_among (contents_value_slots contents)
                 (domain (snd w))
               = true)
          (fun _ _ => True) _ _ _ _ _ indexes index Hjoined i Hin w Hdomain I)).
  - intros w' l [Hc Hv].
    split; apply all_among; intros name Hname;
      [ rewrite <- Hc | rewrite <- Hv ]; exact Hname.
  - intros w' l l' Heq [Hc Hv]. unfold set_of_closures_contents_equal in Heq.
    apply andb_true_iff in Heq as [Heqc Heqv].
    apply (keyed_same_keys_b positive Pos.eqb pos_eqb_decides_equality)
      in Heqc, Heqv.
    split; intros name;
      [ rewrite <- (Hc name); exact (Heqc name)
      | rewrite <- (Hv name); exact (Heqv name) ].
  - intros w' l l' [Hc Hv].
    split; apply all_among; intros name Hname; simpl in Hname;
      rewrite (keyed_among_inter positive Pos.eqb pos_eqb_decides_equality)
        in Hname;
      apply andb_true_iff in Hname as [Hname _];
      [ exact (proj1 (all_among _ _) Hc name Hname)
      | exact (proj1 (all_among _ _) Hv name Hname) ].
  - intros w' l l' [Hc Hv].
    split; apply all_among; intros name Hname; simpl in Hname;
      rewrite (keyed_among_inter positive Pos.eqb pos_eqb_decides_equality)
        in Hname;
      apply andb_true_iff in Hname as [_ Hname];
      [ exact (proj1 (all_among _ _) Hc name Hname)
      | exact (proj1 (all_among _ _) Hv name Hname) ].
  - intros; exact I.
Qed.

Lemma closures_index_bottom
    (indexes : list (row_like_index set_of_closures_contents unit)) :
  n_way_join_index set_of_closures_contents_equal
    set_of_closures_contents_inter (fun _ => Ok tt) indexes
  = Bottom ->
  indexes = [].
Proof. apply n_way_join_index_bottom. intros shapes [=]. Qed.

Lemma closures_entries_sound
    (index : row_like_index set_of_closures_contents unit) :
  step_sound free_variables_of_closures_entry kinds_of_entry
    free_variables_of_closures_entry kinds_of_entry
    (fun statics valuation x valuation' y =>
       ∀ w : closure_and_set,
         closures_index_holds w index ->
         closures_entry_holds statics valuation w x ->
         closures_entry_holds statics valuation' w y)
    (fun s args =>
       (fun env (_ : unit) entries =>
          n_way_join_closures_entry n_way_join_rec env entries)
         s (index_shape index) args).
Proof.
  apply
    (step_sound_bound _ _ _ _
       (all_kept closures_entry_holds closures_entry_holds));
    [ | exact n_way_join_closures_entry_is_sound ].
  intros statics valuation x valuation' y Hbound w _ Hentry.
  exact (Hbound w Hentry).
Qed.

Theorem n_way_join_row_like_for_closures_is_sound :
  step_sound free_variables_of_row_like_for_closures
    kinds_of_closures free_variables_of_row_like_for_closures
    kinds_of_closures (all_kept closures_hold closures_hold)
    (n_way_join_row_like_for_closures join_env_n_way_join_env_extension
       n_way_join_rec).
Proof.
  intros s s' closures answer Hvalid Hwf Hkinded Hdistinct Hargs Hstep.
  unfold n_way_join_row_like_for_closures in Hstep.
  set (knowns :=
    rev
      (map
         (fun closures =>
            (fst closures,
             (row_like_known_closures (snd closures),
              @Or_bottom.Bottom row_like_closures_case)))
         closures)) in Hstep.
  unfold n_way_join_row_like in Hstep.
  pose proof
    (n_way_join_knowns_is_sound
       (fun env (_ : unit) entries =>
          n_way_join_closures_entry n_way_join_rec env entries)
       set_of_closures_contents_equal set_of_closures_contents_inter
       (fun _ => Ok tt) free_variables_of_closures_entry
       kinds_of_entry closures_index_holds closures_entry_holds
       closures_index_sound closures_index_bottom closures_entries_sound)
    as Hknowns_sound.
  match type of Hstep with
  | context [generic_merge_map_known ?eqb ?of_other ?j s knowns] =>
    set (join := j) in Hstep;
    destruct (generic_merge_map_known eqb of_other join s knowns)
      as [[known' sa] | ] eqn:Hmerge; [ | discriminate ]
  end.
  assert (Hjoin_sound :
    case_join_sound closures_case_variables closures_case_aliases
      closures_case_variables closures_case_aliases
      (all_kept closures_case_holds closures_case_holds) join).
  { intros s0 s1 cases result Hwf0 Hkinded0 Hgrows0 Hvalid0 Hdistinct0 Hargs0
      Hjoin.
    unfold join in Hjoin. cbv beta zeta in Hjoin.
    match type of Hjoin with
    | context [n_way_join_knowns ?a ?b ?c ?d ?e s1 ?l] =>
      set (knowns' := l) in Hjoin;
      destruct (n_way_join_knowns a b c d e s1 knowns') as [joined | ]
        eqn:Hknowns; [ | discriminate ]
    end.
    assert (Hdistinct' : join_args_distinct knowns').
    { unfold join_args_distinct, knowns'. rewrite map_map. exact Hdistinct0. }
    assert (Hargs' :
      args_in_scope
        (arg_ok (of_known closures_case_variables)
           (of_known closures_case_aliases))
        s0 knowns').
    { intros id x Hin.
      apply in_map_iff in Hin as ([id' c] & [= <- <-] & Hin).
      exact (Hargs0 id' c Hin). }
    pose proof
      (Hknowns_sound s0 s1 knowns' joined Hwf0 Hkinded0 Hgrows0 Hvalid0
         Hdistinct' Hargs' Hknowns)
      as Hresult.
    destruct joined as [[[c | ] s2] | ]; cbv beta iota in Hjoin;
      [ injection Hjoin as <- | discriminate | injection Hjoin as <- ].
    - destruct Hresult as (Hvalid2 & Hgrows2 & Hk2 & Hholds).
      split; [ exact Hvalid2 | ]. split; [ exact Hgrows2 | ].
      split; [ exact Hk2 | ].
      intros s'' Hgrows''.
      destruct (Hholds s'' Hgrows'') as (Htarget & Hbound & Hkinds).
      split; [ exact Htarget | ]. split; [ | exact Hkinds ].
      intros id c0 use statics valuation valuation' Hin Huse Hsatisfied
        Hextends.
      exact
        (Hbound id (Or_unknown.Known c0) use statics valuation valuation'
           (in_map (fun case => (fst case, Or_unknown.Known (snd case))) cases
              (id, c0) Hin)
           Huse Hsatisfied Hextends).
    - exact (map_eq_nil _ _ Hresult). }
  cbv beta iota delta [n_way_join_case n_way_join_index non_bottom map]
    in Hstep.
  injection Hstep as <- <-.
  assert (Hknowns_distinct : join_args_distinct knowns).
  { unfold join_args_distinct, knowns. rewrite map_rev, map_map.
    apply NoDup_rev. exact Hdistinct. }
  assert (Hin_knowns :
    ∀ (index : Index.t)
      (known_closures : finite_map row_like_closures_case),
      In (index, Row_like_for_closures known_closures) closures ->
      In (index, (known_closures, Or_bottom.Bottom)) knowns).
  { intros index known_closures Hin. unfold knowns. rewrite <- in_rev.
    apply in_map_iff. exists (index, Row_like_for_closures known_closures).
    split; [ reflexivity | exact Hin ]. }
  assert (Hargs_knowns :
    args_in_scope
      (known_in_scope Pos.eqb (fun case : row_like_closures_case => case)
         closures_case_variables closures_case_aliases)
      s knowns).
  { intros index m Hin. unfold knowns in Hin. rewrite <- in_rev in Hin.
    apply in_map_iff in Hin as ([index' [known_closures]] & [= <- <-] & Hin).
    destruct (Hargs index' _ Hin) as (use & Huse & Hvariables & Haliases).
    exists use. split; [ exact Huse | ].
    split; [ | intros o [=] ].
    intros key c Hc. cbn [fst snd row_like_known_closures] in Hc.
    apply (keyed_lookup_in Pos.eqb pos_eqb_decides_equality) in Hc.
    split.
    - intros y Hy. apply Hvariables. simpl. apply in_flat_map.
      exists (key, c). split; [ exact Hc | exact Hy ].
    - intros simple kind Hsimple. apply Haliases. simpl. apply in_or_app.
      right. apply in_flat_map.
      exists (key, c). split; [ exact Hc | exact Hsimple ]. }
  assert (Hknown_shape :
    with_checks = true ->
    Keyed.distinct Pos.eqb (map fst known') = true
    ∧ ∀ (slot : positive) (c : row_like_closures_case),
        Keyed.lookup Pos.eqb slot known' = Some c ->
        closures_case_shape_ok slot c = true).
  { intros Hchecks.
    refine
      (generic_merge_map_known_shape Pos.eqb pos_eqb_decides_equality
         (fun case : row_like_closures_case => case) join
         (fun (slot : positive) (c : row_like_closures_case) =>
            closures_case_shape_ok slot c = true)
         s sa knowns known' Hknowns_distinct _ _ Hmerge).
    - intros slot s1 s2 vals v Hjoin Hvals. unfold join in Hjoin.
      cbv beta zeta in Hjoin.
      match type of Hjoin with
      | context [n_way_join_knowns ?a ?b ?c ?d ?e s1 ?l] =>
        destruct (n_way_join_knowns a b c d e s1 l) as [[[[c0 | ] s3] | ] | ]
          eqn:Hknowns; try discriminate
      end.
      cbv beta iota in Hjoin. injection Hjoin as <- _.
      exact (closures_knowns_join_shape slot s1 s3 vals c0 Hvals Hknowns).
    - intros index m Hin. unfold knowns in Hin. rewrite <- in_rev in Hin.
      apply in_map_iff in Hin as ([index' [known_closures]] & [= <- <-] & Hin).
      cbn [fst snd row_like_known_closures]. split; [ | intros o key [=] ].
      intros slot c Hlookup.
      assert (Hrow_shape :=
        arg_check_passes _ _ s closures index'
          (Row_like_for_closures known_closures) _ Hargs Hin
          (incl_appl _ (incl_refl _)) Hchecks).
      unfold closures_shape_ok in Hrow_shape.
      apply andb_true_iff in Hrow_shape as [_ Hrow_shape].
      apply (keyed_for_all_bindings_b positive Pos.eqb
               pos_eqb_decides_equality) in Hrow_shape.
      exact
        (proj1 (keyed_for_all_bindings_lookup positive Pos.eqb
                  pos_eqb_decides_equality _ _ _)
           Hrow_shape slot c Hlookup). }
  destruct
    (generic_merge_map_known_is_sound Pos.eqb pos_eqb_decides_equality
       (fun case : row_like_closures_case => case) closures_case_variables
       closures_case_aliases
       (all_kept closures_case_holds closures_case_holds) join Hjoin_sound s s
       sa knowns known' Hwf Hkinded (grows_refl s) Hvalid Hknowns_distinct
       Hargs_knowns Hmerge)
    as (Hvalida & Hgrowsa & Hka & Hin_known & Hfound & Hnone).
  split; [ exact Hvalida | ]. split; [ exact Hgrowsa | ].
  split; [ | split ].
  - intros x Hx. simpl in Hx.
    apply in_flat_map in Hx as ([key y] & Hin & Hx). cbn [snd] in Hx.
    destruct (Hfound key y (Hin_known key y Hin)) as (vals & _ & Hholds).
    destruct (Hholds sa (grows_refl sa)) as (Htarget & _ & _).
    exact (Htarget x Hx).
  - intros index [known_closures] use statics valuation valuation' Hin Huse
      Hsatisfied Hextends [[[v slot] functions] value_slots] Hw.
    unfold closures_hold in *.
    apply row_like_for_closures_lookup_iff in Hw as (c & Hc & Hsatisfied_c).
    apply row_like_for_closures_lookup_iff.
    destruct (Keyed.lookup Pos.eqb slot known') as [y | ] eqn:Hy.
    + exists y. split; [ exact Hy | ].
      destruct (Hfound slot y Hy) as (vals & Hvals & Hholds).
      destruct (Hholds sa (grows_refl sa)) as (_ & Hbound & _).
      destruct
        (gather_known_spec Pos.eqb pos_eqb_decides_equality
           (fun case : row_like_closures_case => case) knowns slot vals
           Hknowns_distinct Hvals)
        as (_ & _ & Hto & _).
      destruct
        (Hto (index, (known_closures, Or_bottom.Bottom))
           (Hin_knowns _ _ Hin))
        as (Hsome & _). cbn [fst snd] in Hsome.
      exact
        (Hbound index c use statics valuation valuation' (Hsome c Hc) Huse
           Hsatisfied Hextends (v, slot, functions, value_slots)
           Hsatisfied_c).
    + pose proof
        (gather_known_none Pos.eqb pos_eqb_decides_equality
           (fun case : row_like_closures_case => case) knowns slot
           (Hnone slot Hy) (index, (known_closures, Or_bottom.Bottom))
           (Hin_knowns _ _ Hin))
        as Hv.
      cbn [fst snd] in Hv.
      discriminate (eq_trans (eq_sym Hv) Hc).
  - intros Hk. split; [ exact (Hka Hk) | ].
    intros simple kind Hsimple. simpl in Hsimple.
    apply in_app_or in Hsimple as [Hsimple | Hsimple].
    { refine (answer_check_passes sa _ simple kind _ Hsimple).
      intros Hchecks. destruct (Hknown_shape Hchecks) as [Hdistinct' Hcases'].
      unfold closures_shape_ok. apply andb_true_iff.
      split; [ exact Hdistinct' | ].
      apply (keyed_for_all_bindings_b positive Pos.eqb
               pos_eqb_decides_equality).
      apply (keyed_for_all_bindings_lookup positive Pos.eqb
               pos_eqb_decides_equality).
      intros slot c Hlookup. exact (Hcases' slot c Hlookup). }
    apply in_flat_map in Hsimple as ([key y] & Hin & Hsimple).
    cbn [snd] in Hsimple.
    destruct (Hfound key y (Hin_known key y Hin)) as (vals & _ & Hholds).
    destruct (Hholds sa (grows_refl sa)) as (_ & _ & Hkinds).
    exact (Hkinds Hk simple kind Hsimple).
Qed.

(* ------------------------------------------------------------------------ *)
(* Variants                                                                 *)
(* ------------------------------------------------------------------------ *)

Definition variables_of_extensions (extensions : variant_extensions)
    : list variable :=
  match extensions with
  | No_extensions => []
  | Ext when_immediate when_block =>
    free_variables_of_env_extension when_immediate
    ++ free_variables_of_env_extension when_block
  end.

Definition aliases_of_extensions (extensions : variant_extensions)
    : list (simple * flambda_kind) :=
  match extensions with
  | No_extensions => []
  | Ext when_immediate when_block =>
    kinds_of_extension when_immediate
    ++ kinds_of_extension when_block
  end.

(* What the extensions of a variant say of an immediate, and of a
   block. *)
Definition immediate_extension_holds (statics : static_env) (vars : env)
    (extensions : variant_extensions) : Prop :=
  match extensions with
  | No_extensions => True
  | Ext when_immediate _ => env_extension_satisfied statics vars when_immediate
  end.

Definition block_extension_holds (statics : static_env) (vars : env)
    (extensions : variant_extensions) : Prop :=
  match extensions with
  | No_extensions => True
  | Ext _ when_block => env_extension_satisfied statics vars when_block
  end.

Definition extensions_bound (statics : static_env) (valuation : env)
    (extensions : variant_extensions) (valuation' : env)
    (joined : variant_extensions) : Prop :=
  (immediate_extension_holds statics valuation extensions ->
   immediate_extension_holds statics valuation' joined)
  ∧ (block_extension_holds statics valuation extensions ->
     block_extension_holds statics valuation' joined).

Local Abbreviation blocks_bound :=
  (or_unknown_bound (all_kept blocks_hold blocks_hold)).
Local Abbreviation immediates_bound :=
  (or_unknown_bound (all_kept type_satisfied type_satisfied)).

(* [n_way_join_variant]: each of the three parts answered bounds the
   parts given. *)
Lemma n_way_join_variant_is_sound (s s1 s2 : state)
    (blocks : list (Index.t * Or_unknown.t row_like_for_blocks))
    (imms : list (Index.t * Or_unknown.t flambda_type))
    (extensions : list (Index.t * variant_extensions))
    (answer :
       Or_unknown.t
         (Or_unknown.t row_like_for_blocks * Or_unknown.t flambda_type
          * variant_extensions)) :
  envs_well_formed s -> envs_kinded s -> grows s s1 -> state_valid s1 ->
  join_args_distinct blocks -> join_args_distinct imms ->
  join_args_distinct extensions ->
  args_in_scope
    (arg_ok (of_known free_variables_of_row_like_for_blocks)
       (of_known kinds_of_blocks))
    s blocks ->
  args_in_scope
    (arg_ok (of_known free_variables) (of_known kinds_of_immediates)) s
    imms ->
  args_in_scope (arg_ok variables_of_extensions aliases_of_extensions) s
    extensions ->
  n_way_join_variant import_type join_env_n_way_join_env_extension type_eqb
    block_shape_eqb n_way_join_rec s1 blocks imms extensions
  = Some (answer, s2) ->
  state_valid s2 ∧ grows s1 s2 ∧ (state_kinded s1 -> state_kinded s2)
  ∧ ∀ (blocks' : Or_unknown.t row_like_for_blocks)
      (imms' : Or_unknown.t flambda_type) (extensions' : variant_extensions),
      answer = Or_unknown.Known (blocks', imms', extensions') ->
      answer_holds_from (of_known free_variables_of_row_like_for_blocks)
        (of_known kinds_of_blocks) blocks_bound s1 s2 blocks
        blocks'
      ∧ answer_holds_from (of_known free_variables)
          (of_known kinds_of_immediates) immediates_bound s1 s2 imms imms'
      ∧ answer_holds_from variables_of_extensions aliases_of_extensions
          extensions_bound s1 s2 extensions extensions'.
Proof.
  intros Hwf Hkinded Hgrows Hvalid Hd_blocks Hd_imms Hd_extensions
    Hargs_blocks Hargs_imms Hargs_extensions Hstep.
  unfold n_way_join_variant in Hstep.
  destruct
    (n_way_join_unknown
       (n_way_join_row_like_for_blocks import_type
          join_env_n_way_join_env_extension type_eqb block_shape_eqb
          n_way_join_rec)
       s1 blocks)
    as [[blocks' sa] | ] eqn:Hblocks; [ | discriminate ].
  destruct (n_way_join_unknown n_way_join_rec sa imms) as [[imms' sb] | ]
    eqn:Himms; [ | discriminate ].
  destruct
    (step_sound_at _ _ _ _ _ _
       (n_way_join_unknown_is_sound _ _ _ _ _ _ _ _
          n_way_join_row_like_for_blocks_is_sound)
       s s1 sa blocks blocks' Hwf Hkinded Hgrows Hvalid Hd_blocks
       Hargs_blocks Hblocks)
    as (Hvalida & Hgrowsa & Hka & Hholds_blocks).
  destruct
    (step_sound_at _ _ _ _ _ _
       (n_way_join_unknown_is_sound _ _ _ _ _ _ _ _
          (Hrec_at (Naked_number Kind_naked_immediate)))
       s sa sb imms imms'
       Hwf Hkinded (grows_trans _ _ _ Hgrows Hgrowsa) Hvalida Hd_imms
       Hargs_imms Himms)
    as (Hvalidb & Hgrowsb & Hkb & Hholds_imms).
  (* What is left to do once the extensions are joined. *)
  assert (Hfinal :
    ∀ (joined : variant_extensions) (sc : state),
      state_valid sc -> grows sb sc -> (state_kinded sb -> state_kinded sc) ->
      answer_holds_from variables_of_extensions aliases_of_extensions
        extensions_bound sb sc extensions joined ->
      match blocks', imms', joined with
      | Or_unknown.Unknown, Or_unknown.Unknown, No_extensions =>
        Some (Or_unknown.Unknown, sc)
      | _, _, _ => Some (Or_unknown.Known (blocks', imms', joined), sc)
      end
      = Some (answer, s2) ->
      state_valid s2 ∧ grows s1 s2 ∧ (state_kinded s1 -> state_kinded s2)
      ∧ ∀ (blocks'' : Or_unknown.t row_like_for_blocks)
          (imms'' : Or_unknown.t flambda_type)
          (extensions'' : variant_extensions),
          answer = Or_unknown.Known (blocks'', imms'', extensions'') ->
          answer_holds_from (of_known free_variables_of_row_like_for_blocks)
            (of_known kinds_of_blocks) blocks_bound s1 s2
            blocks blocks''
          ∧ answer_holds_from (of_known free_variables)
              (of_known kinds_of_immediates) immediates_bound s1 s2 imms
              imms''
          ∧ answer_holds_from variables_of_extensions aliases_of_extensions
              extensions_bound s1 s2 extensions extensions'').
  { intros joined sc Hvalidc Hgrowsc Hkc Hholds_extensions Hanswer.
    assert (Hs2 : sc = s2).
    { destruct blocks', imms', joined; cbv beta iota in Hanswer; congruence. }
    subst sc.
    split; [ exact Hvalidc | ].
    split;
      [ exact (grows_trans _ _ _ Hgrowsa (grows_trans _ _ _ Hgrowsb Hgrowsc))
      | ].
    split; [ auto | ].
    intros blocks'' imms'' extensions'' ->.
    assert (Heq : blocks'' = blocks' ∧ imms'' = imms' ∧ extensions'' = joined).
    { destruct blocks', imms', joined; cbv beta iota in Hanswer;
        repeat split; congruence. }
    destruct Heq as (-> & -> & ->).
    split; [ | split ].
    - exact
        (answer_holds_from_later _ _ _ s1 s1 sa s2 blocks blocks'
           (fun H => H) (grows_trans _ _ _ Hgrowsb Hgrowsc) Hholds_blocks).
    - exact
        (answer_holds_from_later _ _ _ s1 sa sb s2 imms imms' Hka Hgrowsc
           Hholds_imms).
    - exact
        (answer_holds_from_later _ _ _ s1 sb s2 s2 extensions joined
           (fun H => Hkb (Hka H)) (grows_refl s2) Hholds_extensions). }
  (* No extensions. *)
  assert (Hnone :
    answer_holds_from variables_of_extensions aliases_of_extensions
      extensions_bound sb sb extensions No_extensions).
  { intros s' _. split; [ intros x [] | ].
    split; [ | intros _ simple kind [] ].
    intros index e use statics valuation valuation' _ _ _ _.
    split; intros _; exact I. }
  destruct (gather_extensions [] [] extensions)
    as [[when_immediates when_blocks] | ] eqn:Hgather.
  2: { exact
         (Hfinal No_extensions sb Hvalidb (grows_refl sb) (fun H => H) Hnone
            Hstep). }
  cbv zeta in Hstep.
  destruct
    (n_way_join_env_extension join_env_n_way_join_env_extension sb
       when_immediates)
    as [[when_immediate sc] | ] eqn:Hwhen_immediate; [ | discriminate ].
  destruct
    (n_way_join_env_extension join_env_n_way_join_env_extension sc
       when_blocks)
    as [[when_block sd] | ] eqn:Hwhen_block; [ | discriminate ].
  destruct
    (env_extension_is_empty when_immediate
     && env_extension_is_empty when_block).
  { exact
      (Hfinal No_extensions sb Hvalidb (grows_refl sb) (fun H => H) Hnone
         Hstep). }
  destruct (gather_extensions_spec extensions [] [] _ _ Hgather)
    as (Hfst_i & Hfst_b & Hto & Hfrom_i & Hfrom_b & _ & _).
  rewrite app_nil_r in Hfst_i, Hfst_b.
  assert (Hd_i : join_args_distinct when_immediates).
  { unfold join_args_distinct. rewrite Hfst_i. apply NoDup_rev.
    exact Hd_extensions. }
  assert (Hd_b : join_args_distinct when_blocks).
  { unfold join_args_distinct. rewrite Hfst_b. apply NoDup_rev.
    exact Hd_extensions. }
  assert (Hargs_i :
    args_in_scope
      (arg_ok free_variables_of_env_extension kinds_of_extension) s
      when_immediates).
  { apply (args_of_parts _ _ _ _ s extensions when_immediates Hargs_extensions).
    intros index wi Hin. destruct (Hfrom_i index wi Hin) as [[] | [wb Hin']].
    exists (Ext wi wb). split; [ exact Hin' | ].
    split; apply incl_appl, incl_refl. }
  assert (Hargs_b :
    args_in_scope
      (arg_ok free_variables_of_env_extension kinds_of_extension) s
      when_blocks).
  { apply (args_of_parts _ _ _ _ s extensions when_blocks Hargs_extensions).
    intros index wb Hin. destruct (Hfrom_b index wb Hin) as [[] | [wi Hin']].
    exists (Ext wi wb). split; [ exact Hin' | ].
    split; apply incl_appr, incl_refl. }
  assert (Hgrows_sb : grows s sb).
  { exact (grows_trans _ _ _ Hgrows (grows_trans _ _ _ Hgrowsa Hgrowsb)). }
  destruct
    (step_sound_at _ _ _ _ _ _
       Hextension_kinds
       s sb sc when_immediates when_immediate Hwf Hkinded Hgrows_sb Hvalidb
       Hd_i Hargs_i Hwhen_immediate)
    as (Hvalidc & Hgrowsc & Hkc & Hholds_i).
  destruct
    (step_sound_at _ _ _ _ _ _
       Hextension_kinds
       s sc sd when_blocks when_block Hwf Hkinded
       (grows_trans _ _ _ Hgrows_sb Hgrowsc) Hvalidc Hd_b Hargs_b Hwhen_block)
    as (Hvalidd & Hgrowsd & Hkd & Hholds_b).
  apply
    (Hfinal (Ext when_immediate when_block) sd Hvalidd
       (grows_trans _ _ _ Hgrowsc Hgrowsd) (fun H => Hkd (Hkc H)));
    [ | exact Hstep ].
  intros s' Hgrows'.
  destruct (Hholds_i s' (grows_trans _ _ _ Hgrowsd Hgrows'))
    as (Htarget_i & Hbound_i & Hkinds_i).
  destruct (Hholds_b s' Hgrows') as (Htarget_b & Hbound_b & Hkinds_b).
  split; [ | split ].
  - intros x Hx. simpl in Hx. apply in_app_or in Hx as [Hx | Hx];
      [ exact (Htarget_i x Hx) | exact (Htarget_b x Hx) ].
  - intros index e use statics valuation valuation' Hin Huse Hsatisfied
      Hextends.
    destruct (Hto index e Hin) as (wi & wb & -> & Hin_i & Hin_b).
    split; simpl.
    + exact
        (Hbound_i index wi use statics valuation valuation' Hin_i Huse
           Hsatisfied Hextends).
    + exact
        (Hbound_b index wb use statics valuation valuation' Hin_b Huse
           Hsatisfied Hextends).
  - intros Hk simple kind Hsimple. simpl in Hsimple.
    apply in_app_or in Hsimple as [Hsimple | Hsimple];
      [ exact (Hkinds_i Hk simple kind Hsimple)
      | exact (Hkinds_b (Hkc Hk) simple kind Hsimple) ].
Qed.

(* ------------------------------------------------------------------------ *)
(* Heads of kind value that are not null                                    *)
(* ------------------------------------------------------------------------ *)

(* The variable joined for [Is_int], and the one for [Get_tag]: it
   holds in the target what the [Simple] deduced for a variant holds in
   its input; there is none if a variant has no such [Simple]. *)
Definition is_int_bound (statics : static_env) (valuation : env)
    (fields : variant_fields) (valuation' : env) (x : option variable)
    : Prop :=
  match is_int_simple_of fields with
  | Some simple =>
    ∀ n : Z,
      eval_simple valuation simple = Some (Value_const (Naked_immediate n)) ->
      variable_is valuation' x n
  | None => x = None
  end.

Definition get_tag_bound (statics : static_env) (valuation : env)
    (fields : variant_fields) (valuation' : env) (x : option variable)
    : Prop :=
  match get_tag_simple_of fields with
  | Ok simple =>
    ∀ n : Z,
      eval_simple valuation simple = Some (Value_const (Naked_immediate n)) ->
      variable_is valuation' x n
  | Bottom => True
  | Unknown => x = None
  end.

(* [n_way_join_relation_simples] on the [Simple]s gathered of things
   [args], if any were. *)
Lemma relation_simples_step {X : Type}
    (bound : static_env -> env -> X -> env -> option variable -> Prop)
    (s s1 s2 : state) (args : list (Index.t * X))
    (simples_opt : option (list (Index.t * simple))) (x : option variable) :
  envs_well_formed s -> envs_kinded s -> grows s s1 -> state_valid s1 ->
  (∀ simples : list (Index.t * simple),
     simples_opt = Some simples ->
     join_args_distinct simples
     ∧ args_in_scope
         (arg_ok (fun _ : simple => [])
            (fun simple => [(simple, Naked_number Kind_naked_immediate)]))
         s simples) ->
  (simples_opt = None ->
   ∀ (statics : static_env) (valuation : env) (a : X) (valuation' : env),
     bound statics valuation a valuation' None) ->
  (∀ simples : list (Index.t * simple),
     simples_opt = Some simples ->
     ∀ (index : Index.t) (a : X) (statics : static_env)
       (valuation valuation' : env) (y : option variable),
       In (index, a) args ->
       (∀ simple : simple,
          In (index, simple) simples ->
          ∀ n : Z,
            eval_simple valuation simple
            = Some (Value_const (Naked_immediate n)) ->
            variable_is valuation' y n) ->
       bound statics valuation a valuation' y) ->
  n_way_join_relation_simples n_way_join_simples s1 simples_opt
  = Some (x, s2) ->
  state_valid s2 ∧ grows s1 s2 ∧ (state_kinded s1 -> state_kinded s2)
  ∧ answer_holds_from free_variables_of_option aliases_of_relation bound s1
      s2 args x.
Proof.
  intros Hwf Hkinded Hgrows Hvalid Hsome Hnone Hbound Hstep.
  destruct simples_opt as [simples | ].
  - destruct (Hsome simples eq_refl) as (Hdistinct & Hargs).
    destruct
      (step_sound_at _ _ _ _ _ _
         (n_way_join_relation_simples_is_sound n_way_join_simples Hsimples) s
         s1 s2 simples x Hwf Hkinded Hgrows Hvalid Hdistinct Hargs Hstep)
      as (Hvalid2 & Hgrows2 & Hk2 & Hholds).
    split; [ exact Hvalid2 | ]. split; [ exact Hgrows2 | ].
    split; [ exact Hk2 | ].
    intros s' Hgrows'.
    destruct (Hholds s' Hgrows') as (Htarget & Hbound' & Hkinds).
    split; [ exact Htarget | ]. split; [ | exact Hkinds ].
    intros index a use statics valuation valuation' Hin Huse Hsatisfied
      Hextends.
    apply (Hbound simples eq_refl index a statics valuation valuation' x Hin).
    intros simple Hin' n Heval.
    exact
      (Hbound' index simple use statics valuation valuation' Hin' Huse
         Hsatisfied Hextends n Heval).
  - injection Hstep as <- <-.
    split; [ exact Hvalid | ]. split; [ apply grows_refl | ].
    split; [ auto | ].
    intros s' _. split; [ intros y [] | ].
    split; [ | intros _ simple kind [] ].
    intros index a use statics valuation valuation' _ _ _ _.
    exact (Hnone eq_refl statics valuation a valuation').
Qed.

Local Abbreviation non_null_bound :=
  (known_bound
     (all_kept head_of_kind_value_non_null_satisfied
        head_of_kind_value_non_null_satisfied)).

Theorem n_way_join_head_of_kind_value_non_null_is_sound :
  step_sound free_variables_of_head_of_kind_value_non_null
    kinds_of_non_null
    (of_known free_variables_of_head_of_kind_value_non_null)
    (of_known kinds_of_non_null) non_null_bound
    (n_way_join_head_of_kind_value_non_null n_way_join_simples import_type
       join_env_n_way_join_env_extension type_eqb block_shape_eqb
       n_way_join_rec).
Proof.
  intros s s' heads answer Hvalid Hwf Hkinded Hdistinct Hargs Hstep.
  unfold n_way_join_head_of_kind_value_non_null in Hstep.
  destruct heads as [ | [first_id first_head] other_heads];
    [ discriminate | ].
  set (heads := (first_id, first_head) :: other_heads) in *.
  (* The answer that says nothing. *)
  assert (Hunknown :
    ∀ s1 : state,
      state_valid s1 -> grows s s1 -> (state_kinded s -> state_kinded s1) ->
      Some (@Or_unknown.Unknown head_of_kind_value_non_null, s1)
      = Some (answer, s') ->
      state_valid s' ∧ grows s s'
      ∧ in_target (of_known free_variables_of_head_of_kind_value_non_null) s'
          answer
      ∧ (∀ (index : Index.t) (x : head_of_kind_value_non_null)
           (use : Typing_env.t) (statics : static_env)
           (valuation valuation' : env),
           In (index, x) heads ->
           joined_env s' index = Some use ->
           typing_env_satisfied_strictly statics valuation use ->
           extends s' index valuation valuation' ->
           non_null_bound statics valuation x valuation' answer)
      ∧ (state_kinded s ->
         state_kinded s'
         ∧ kinded_in_target
             (of_known kinds_of_non_null) s' answer)).
  { intros s1 Hvalid1 Hgrows1 Hk1 [= <- <-].
    split; [ exact Hvalid1 | ]. split; [ exact Hgrows1 | ].
    split; [ intros x [] | ]. split; [ intros; exact I | ].
    intros Hk. split; [ exact (Hk1 Hk) | intros simple kind [] ]. }
  destruct first_head
    as [is_int immediates get_tag blocks extensions is_unique | alloc_mode
       | k n alloc_mode | by_function_slot alloc_mode | strs].
  - (* Variants. *)
    set (first :=
      {| variant_is_int := is_int;
         variant_immediates := immediates;
         variant_get_tag := get_tag;
         variant_blocks := blocks;
         variant_extensions_of := extensions;
         variant_is_unique := is_unique |}) in Hstep.
    cbv zeta in Hstep. rewrite gather_heads_eq in Hstep.
    destruct (indexed_map_opt (fun _ head => read_variant head) other_heads)
      as [l | ] eqn:Hl; cbn [option_map] in Hstep;
      [ | exact (Hunknown s Hvalid (grows_refl s) (fun H => H) Hstep) ].
    change
      (List.fold_right (fun x acc => add_variant_parts (fst x) (snd x) acc)
         (first_variant_parts first_id first) l)
      with (variant_parts_of first_id first l) in Hstep.
    set (parts := variant_parts_of first_id first l) in *.
    set (all := l ++ [(first_id, first)]).
    destruct (variant_parts_spec first_id first l)
      as (Hblocks & Himmediates & Hextensions & His_int & Hget_tag).
    fold parts all in Hblocks, Himmediates, Hextensions, His_int, Hget_tag.
    destruct (indexed_map_opt_spec _ _ l Hl) as (Hfst & Hto & Hfrom).
    assert (Hall_distinct : NoDup (map fst all)).
    { unfold all. rewrite map_app. cbn [map fst]. rewrite Hfst.
      apply snoc_distinct. exact Hdistinct. }
    assert (Hall_from :
      ∀ (index : Index.t) (fields : variant_fields),
        In (index, fields) all -> In (index, variant_of fields) heads).
    { intros index fields Hin.
      apply in_app_or in Hin as [Hin | [[= <- <-] | []]].
      - destruct (Hfrom index fields Hin) as (head & Hread & Hin').
        right. rewrite (read_variant_eq head fields Hread) in Hin'.
        exact Hin'.
      - left. reflexivity. }
    assert (Hall_to :
      ∀ (index : Index.t) (head : head_of_kind_value_non_null),
        In (index, head) heads ->
        ∃ fields : variant_fields,
          head = variant_of fields ∧ In (index, fields) all).
    { intros index head [[= <- <-] | Hin].
      - exists first. split; [ reflexivity | ].
        apply in_or_app. right. left. reflexivity.
      - destruct (Hto index head Hin) as (fields & Hread & Hin').
        exists fields. split; [ exact (read_variant_eq head fields Hread) | ].
        apply in_or_app. left. exact Hin'. }
    assert (Hall_scope :
      args_in_scope
        (arg_ok
           (fun fields =>
              free_variables_of_head_of_kind_value_non_null
                (variant_of fields))
           (fun fields =>
              kinds_of_non_null (variant_of fields)))
        s all).
    { intros index fields Hin.
      exact (Hargs index (variant_of fields) (Hall_from index fields Hin)). }
    (* The relation variables. *)
    destruct (n_way_join_relation_simples n_way_join_simples s
                (parts_is_int parts))
      as [[is_int' sa] | ] eqn:Hjoin_is_int; [ | discriminate ].
    destruct (n_way_join_relation_simples n_way_join_simples sa
                (parts_get_tag parts))
      as [[get_tag' sb] | ] eqn:Hjoin_get_tag; [ | discriminate ].
    destruct
      (relation_simples_step is_int_bound s s sa all (parts_is_int parts)
         is_int' Hwf Hkinded (grows_refl s) Hvalid)
      as (Hvalida & Hgrowsa & Hka & Hholds_is_int);
      [ | | | exact Hjoin_is_int | ].
    { intros simples Hsimples'.
      destruct (indexed_map_opt_spec _ _ simples (His_int simples Hsimples'))
        as (Hfst' & _ & Hfrom').
      split;
        [ unfold join_args_distinct; rewrite Hfst'; exact Hall_distinct | ].
      intros index simple Hin.
      destruct (Hfrom' index simple Hin) as (fields & Hsimple & Hin').
      destruct (Hall_scope index fields Hin') as (use & Huse & _ & Haliases).
      exists use. split; [ exact Huse | ]. split; [ intros y [] | ].
      intros simple' kind [[= <- <-] | []].
      destruct (deduce_is_int_simple_cases _ _ _ _ Hsimple)
        as [[b ->] | (var & Hvar & ->)]; [ reflexivity | ].
      apply Haliases. unfold variant_of. simpl. rewrite Hvar. left.
      reflexivity. }
    { intros Hnone' statics valuation fields valuation'.
      unfold is_int_bound. destruct (is_int_simple_of fields);
        [ intros; exact I | reflexivity ]. }
    { intros simples Hsimples' index fields statics valuation valuation' y
        Hin Hy.
      destruct (indexed_map_opt_spec _ _ simples (His_int simples Hsimples'))
        as (_ & Hto' & _).
      destruct (Hto' index fields Hin) as (simple & Hsimple & Hin').
      unfold is_int_bound. rewrite Hsimple. exact (Hy simple Hin'). }
    destruct
      (relation_simples_step get_tag_bound s sa sb all (parts_get_tag parts)
         get_tag' Hwf Hkinded Hgrowsa Hvalida)
      as (Hvalidb & Hgrowsb & Hkb & Hholds_get_tag);
      [ | | | exact Hjoin_get_tag | ].
    { intros simples Hsimples'.
      destruct (Hget_tag simples Hsimples') as (-> & _).
      split.
      - apply flat_map_indexed_distinct; [ | | exact Hall_distinct ].
        + intros [id fields] index simple. unfold get_tag_simples_of.
          destruct (get_tag_simple_of (snd (id, fields)));
            [ intros [] | intros [[= <- _] | []]; reflexivity | intros [] ].
        + intros [id fields]. unfold get_tag_simples_of.
          destruct (get_tag_simple_of (snd (id, fields)));
            try apply NoDup_nil.
          apply NoDup_cons; [ intros [] | apply NoDup_nil ].
      - intros index simple Hin.
        apply in_flat_map in Hin as ([id fields] & Hin' & Hsimple).
        unfold get_tag_simples_of in Hsimple. cbn [fst snd] in Hsimple.
        destruct (get_tag_simple_of fields) as [ | simple' | ] eqn:Htag;
          [ destruct Hsimple | | destruct Hsimple ].
        destruct Hsimple as [[= <- <-] | []].
        destruct (Hall_scope id fields Hin') as (use & Huse & _ & Haliases).
        exists use. split; [ exact Huse | ]. split; [ intros y [] | ].
        intros simple'' kind [[= <- <-] | []].
        destruct (deduce_get_tag_simple_cases _ _ _ Htag)
          as [[tag ->] | (var & Hvar & ->)]; [ reflexivity | ].
        apply Haliases. unfold variant_of. simpl. rewrite Hvar.
        apply in_or_app. right. apply in_or_app. right. left. reflexivity. }
    { intros Hnone' statics valuation fields valuation'.
      unfold get_tag_bound. destruct (get_tag_simple_of fields);
        [ reflexivity | intros; exact I | exact I ]. }
    { intros simples Hsimples' index fields statics valuation valuation' y
        Hin Hy.
      destruct (Hget_tag simples Hsimples') as (-> & Hknown).
      unfold get_tag_bound.
      destruct (get_tag_simple_of fields) as [ | simple | ] eqn:Htag;
        [ | | exact I ].
      - destruct (Hknown (index, fields) Hin Htag).
      - apply Hy. apply in_flat_map. exists (index, fields).
        split; [ exact Hin | ]. unfold get_tag_simples_of. cbn [fst snd].
        rewrite Htag. left. reflexivity. }
    (* The parts. *)
    rewrite Hblocks, Himmediates, Hextensions in Hstep.
    unfold map_join_result in Hstep.
    match type of Hstep with
    | context [n_way_join_variant ?a ?b ?c ?d ?e sb ?bs0 ?is0 ?es0] =>
      set (blocks_given := bs0) in Hstep;
      set (imms_given := is0) in Hstep;
      set (extensions_given := es0) in Hstep;
      destruct (n_way_join_variant a b c d e sb blocks_given imms_given
                  extensions_given)
        as [[joined sc] | ] eqn:Hjoin_variant; [ | discriminate ]
    end.
    injection Hstep as <- <-.
    assert (Hgrows_sb : grows s sb)
      by exact (grows_trans _ _ _ Hgrowsa Hgrowsb).
    destruct
      (n_way_join_variant_is_sound s sb sc blocks_given imms_given
         extensions_given joined Hwf Hkinded Hgrows_sb Hvalidb)
      as (Hvalidc & Hgrowsc & Hkc & Hjoined);
      [ | | | | | | exact Hjoin_variant | ].
    { unfold join_args_distinct, blocks_given. rewrite map_map.
      exact Hall_distinct. }
    { unfold join_args_distinct, imms_given. rewrite map_map.
      exact Hall_distinct. }
    { unfold join_args_distinct, extensions_given. rewrite map_map.
      exact Hall_distinct. }
    { apply (args_of_parts _ _ _ _ s all blocks_given Hall_scope).
      intros index b Hin.
      apply in_map_iff in Hin as ([id fields] & [= <- <-] & Hin).
      exists fields. split; [ exact Hin | ]. unfold variant_of. simpl.
      split; apply incl_appr, incl_appr, incl_appr, incl_appl, incl_refl. }
    { apply (args_of_parts _ _ _ _ s all imms_given Hall_scope).
      intros index b Hin.
      apply in_map_iff in Hin as ([id fields] & [= <- <-] & Hin).
      exists fields. split; [ exact Hin | ]. unfold variant_of. simpl.
      split; apply incl_appr, incl_appl, incl_refl. }
    { apply (args_of_parts _ _ _ _ s all extensions_given Hall_scope).
      intros index b Hin.
      apply in_map_iff in Hin as ([id fields] & [= <- <-] & Hin).
      exists fields. split; [ exact Hin | ]. unfold variant_of. simpl.
      split; apply incl_appr, incl_appr, incl_appr, incl_appr, incl_refl. }
    destruct joined as [[[blocks' imms'] extensions'] | ].
    2: { split; [ exact Hvalidc | ].
         split; [ exact (grows_trans _ _ _ Hgrows_sb Hgrowsc) | ].
         split; [ intros x [] | ]. split; [ intros; exact I | ].
         intros Hk. split; [ exact (Hkc (Hkb (Hka Hk))) | ].
         intros simple kind []. }
    destruct (Hjoined blocks' imms' extensions' eq_refl)
      as (Hholds_blocks & Hholds_imms & Hholds_extensions).
    destruct (Hholds_is_int sc (grows_trans _ _ _ Hgrowsb Hgrowsc))
      as (Htarget_is_int & Hbound_is_int & Hkinds_is_int).
    destruct (Hholds_get_tag sc Hgrowsc)
      as (Htarget_get_tag & Hbound_get_tag & Hkinds_get_tag).
    destruct (Hholds_blocks sc (grows_refl sc))
      as (Htarget_blocks & Hbound_blocks & Hkinds_blocks).
    destruct (Hholds_imms sc (grows_refl sc))
      as (Htarget_imms & Hbound_imms & Hkinds_imms).
    destruct (Hholds_extensions sc (grows_refl sc))
      as (Htarget_extensions & Hbound_extensions & Hkinds_extensions).
    split; [ exact Hvalidc | ].
    split; [ exact (grows_trans _ _ _ Hgrows_sb Hgrowsc) | ].
    split; [ | split ].
    + intros x Hx. simpl in Hx.
      apply in_app_or in Hx as [Hx | Hx]; [ exact (Htarget_is_int x Hx) | ].
      apply in_app_or in Hx as [Hx | Hx]; [ exact (Htarget_imms x Hx) | ].
      apply in_app_or in Hx as [Hx | Hx]; [ exact (Htarget_get_tag x Hx) | ].
      apply in_app_or in Hx as [Hx | Hx];
        [ exact (Htarget_blocks x Hx) | exact (Htarget_extensions x Hx) ].
    + intros index head use statics valuation valuation' Hin Huse Hsatisfied
        Hextends v Hv.
      destruct (Hall_to index head Hin) as (fields & -> & Hin').
      pose proof
        (Hbound_is_int index fields use statics valuation valuation' Hin'
           Huse Hsatisfied Hextends)
        as His_int_b.
      pose proof
        (Hbound_get_tag index fields use statics valuation valuation' Hin'
           Huse Hsatisfied Hextends)
        as Hget_tag_b.
      assert (Hblocks_b :
        blocks_bound statics valuation (variant_blocks fields) valuation'
          blocks').
      { apply (Hbound_blocks index _ use statics valuation valuation');
          [ | exact Huse | exact Hsatisfied | exact Hextends ].
        apply in_map_iff. exists (index, fields).
        split; [ reflexivity | exact Hin' ]. }
      assert (Himms_b :
        immediates_bound statics valuation (variant_immediates fields)
          valuation' imms').
      { apply (Hbound_imms index _ use statics valuation valuation');
          [ | exact Huse | exact Hsatisfied | exact Hextends ].
        apply in_map_iff. exists (index, fields).
        split; [ reflexivity | exact Hin' ]. }
      assert (Hextensions_b :
        extensions_bound statics valuation (variant_extensions_of fields)
          valuation' extensions').
      { apply (Hbound_extensions index _ use statics valuation valuation');
          [ | exact Huse | exact Hsatisfied | exact Hextends ].
        apply in_map_iff. exists (index, fields).
        split; [ reflexivity | exact Hin' ]. }
      unfold is_int_bound, is_int_simple_of in His_int_b.
      unfold get_tag_bound, get_tag_simple_of in Hget_tag_b.
      pose proof Hv as Hv'. unfold variant_of in Hv'.
      destruct Hv'
        as [(i & -> & His_int_i & Himm & Hext)
           | (tag & fs & Hblock & His_int_i & Hget_tag_i & Hrow & Hext)].
      * left. exists i. split; [ reflexivity | ].
        split.
        { destruct (deduce_is_int_simple _ _ _) as [simple | ] eqn:Hsimple;
            [ | rewrite His_int_b; exact I ].
          apply His_int_b.
          destruct
            (deduce_is_int_simple_is_sound statics valuation _ _ _ _ _ _ _
               simple Hv Hsimple)
            as [[_ Heval] | [[tag [fs Hblock]] _]]; [ exact Heval | ].
          destruct (block_not_immediate statics i tag fs Hblock). }
        split.
        { destruct imms' as [ty' | ]; [ | exact I ].
          destruct (variant_immediates fields) as [ty | ];
            [ exact (Himms_b _ Himm) | destruct Himms_b ]. }
        exact (proj1 Hextensions_b Hext).
      * right. exists tag, fs. split; [ exact Hblock | ].
        split.
        { destruct (deduce_is_int_simple _ _ _) as [simple | ] eqn:Hsimple;
            [ | rewrite His_int_b; exact I ].
          apply His_int_b.
          destruct
            (deduce_is_int_simple_is_sound statics valuation _ _ _ _ _ _ _
               simple Hv Hsimple)
            as [[[i ->] _] | [_ Heval]]; [ | exact Heval ].
          destruct (block_not_immediate statics i tag fs Hblock). }
        split.
        { pose proof
            (deduce_get_tag_simple_is_sound statics valuation _ _ _ _ _ _ _
               tag fs Hv Hblock)
            as Hdeduce.
          destruct (deduce_get_tag_simple _ _) as [ | simple | ];
            [ rewrite Hget_tag_b; exact I | exact (Hget_tag_b tag Hdeduce)
            | destruct Hdeduce ]. }
        split.
        { destruct blocks' as [b' | ]; [ | exact I ].
          destruct (variant_blocks fields) as [b | ];
            [ exact (Hblocks_b (tag, fs) Hrow) | destruct Hblocks_b ]. }
        exact (proj2 Hextensions_b Hext).
    + intros Hk. split; [ exact (Hkc (Hkb (Hka Hk))) | ].
      intros simple kind Hsimple. simpl in Hsimple.
      apply in_app_or in Hsimple as [Hsimple | Hsimple];
        [ exact (Hkinds_is_int Hk simple kind Hsimple) | ].
      apply in_app_or in Hsimple as [Hsimple | Hsimple];
        [ exact (Hkinds_imms (Hkb (Hka Hk)) simple kind Hsimple) | ].
      apply in_app_or in Hsimple as [Hsimple | Hsimple];
        [ exact (Hkinds_get_tag (Hka Hk) simple kind Hsimple) | ].
      apply in_app_or in Hsimple as [Hsimple | Hsimple];
        [ exact (Hkinds_blocks (Hkb (Hka Hk)) simple kind Hsimple)
        | exact (Hkinds_extensions (Hkb (Hka Hk)) simple kind Hsimple) ].
  - (* Mutable blocks: no value is one. *)
    rewrite gather_heads_eq in Hstep.
    destruct
      (indexed_map_opt (fun _ head => read_mutable_block head) other_heads)
      as [l | ] eqn:Hl; cbn [option_map] in Hstep;
      [ | exact (Hunknown s Hvalid (grows_refl s) (fun H => H) Hstep) ].
    injection Hstep as <- <-.
    destruct (indexed_map_opt_spec _ _ l Hl) as (_ & Hto & _).
    split; [ exact Hvalid | ]. split; [ apply grows_refl | ].
    split; [ intros x [] | ].
    split; [ | intros Hk; split; [ exact Hk | intros simple kind [] ] ].
    intros index head use statics valuation valuation' Hin _ _ _ v Hv.
    destruct Hin as [[= <- <-] | Hin]; [ destruct Hv | ].
    destruct (Hto index head Hin) as (mode & Hread & _).
    destruct head; try discriminate. destruct Hv.
  - (* Boxed numbers. *)
    rewrite gather_heads_eq in Hstep.
    destruct
      (indexed_map_opt (fun _ head => read_boxed_number k head) other_heads)
      as [l | ] eqn:Hl; cbn [option_map] in Hstep;
      [ | exact (Hunknown s Hvalid (grows_refl s) (fun H => H) Hstep) ].
    match type of Hstep with
    | context [List.fold_right ?f ?i l] =>
      destruct (List.fold_right f i l) as [ns mode] eqn:Hgathered
    end.
    assert (Hns :
      ns = map (fun x => (fst x, fst (snd x))) l ++ [(first_id, n)]).
    { change ns with (fst (ns, mode)). rewrite <- Hgathered.
      exact (gathered_pairs_fst join_alloc_mode l [(first_id, n)] alloc_mode). }
    destruct (gathered_parts (read_boxed_number k) first_id n other_heads l
                Hdistinct Hl)
      as (Hd_ns & Hto & Hfirst & Hfrom).
    rewrite <- Hns in Hd_ns, Hto, Hfirst, Hfrom. clear Hns Hgathered.
    unfold map_join_result in Hstep.
    destruct (n_way_join_rec s ns) as [[joined s1] | ] eqn:Hjoin;
      [ | discriminate ].
    injection Hstep as <- <-.
    assert (Hargs_ns :
      args_in_scope
        (arg_ok free_variables (kinds_of_type_at (unboxed_kind k))) s ns).
    { apply (args_of_parts _ _ _ _ s heads ns Hargs). intros index ty Hin.
      destruct (Hfrom index ty Hin)
        as [[-> ->] | (head & mode' & Hin' & Hread)].
      - exists (Boxed_number k n alloc_mode).
        split; [ left; reflexivity | split; apply incl_refl ].
      - exists head. split; [ right; exact Hin' | ].
        rewrite (read_boxed_number_eq k head ty mode' Hread).
        split; apply incl_refl. }
    destruct
      (step_sound_at _ _ _ _ _ _ (Hrec_at (unboxed_kind k)) s s s1 ns joined
         Hwf Hkinded (grows_refl s) Hvalid Hd_ns Hargs_ns Hjoin)
      as (Hvalid1 & Hgrows1 & Hk1 & Hholds).
    destruct (Hholds s1 (grows_refl s1)) as (Htarget & Hbound & Hkinds).
    split; [ exact Hvalid1 | ]. split; [ exact Hgrows1 | ].
    split; [ | split ].
    + destruct joined as [ty | ]; [ exact Htarget | intros x [] ].
    + intros index head use statics valuation valuation' Hin Huse Hsatisfied
        Hextends.
      destruct joined as [ty | ]; [ | exact I ].
      intros v Hv.
      assert (Hhead :
        ∃ (ty_i : flambda_type) (mode_i : Alloc_mode.t),
          head = Boxed_number k ty_i mode_i ∧ In (index, ty_i) ns).
      { destruct Hin as [[= <- <-] | Hin].
        - exists n, alloc_mode. split; [ reflexivity | exact Hfirst ].
        - destruct (Hto index head Hin) as (ty_i & mode_i & Hread & Hin').
          exists ty_i, mode_i.
          split; [ exact (read_boxed_number_eq k head ty_i mode_i Hread) | ].
          exact Hin'. }
      destruct Hhead as (ty_i & mode_i & -> & Hin_i).
      destruct Hv as (c & Hboxed & Hkind & Hc). exists c.
      split; [ exact Hboxed | ]. split; [ exact Hkind | ].
      exact
        (Hbound index ty_i use statics valuation valuation' Hin_i Huse
           Hsatisfied Hextends (Value_const c) Hc).
    + intros Hk. split; [ exact (Hk1 Hk) | ].
      destruct joined as [ty | ];
        [ exact (Hkinds Hk) | intros simple kind [] ].
  - (* Closures. *)
    rewrite gather_heads_eq in Hstep.
    destruct (indexed_map_opt (fun _ head => read_closures head) other_heads)
      as [l | ] eqn:Hl; cbn [option_map] in Hstep;
      [ | exact (Hunknown s Hvalid (grows_refl s) (fun H => H) Hstep) ].
    match type of Hstep with
    | context [List.fold_right ?f ?i l] =>
      destruct (List.fold_right f i l) as [function_slots mode] eqn:Hgathered
    end.
    assert (Hslots :
      function_slots
      = map (fun x => (fst x, fst (snd x))) l
        ++ [(first_id, by_function_slot)]).
    { change function_slots with (fst (function_slots, mode)).
      rewrite <- Hgathered.
      exact
        (gathered_pairs_fst join_alloc_mode l [(first_id, by_function_slot)]
           alloc_mode). }
    destruct (gathered_parts read_closures first_id by_function_slot
                other_heads l Hdistinct Hl)
      as (Hd_slots & Hto & Hfirst & Hfrom).
    rewrite <- Hslots in Hd_slots, Hto, Hfirst, Hfrom. clear Hslots Hgathered.
    destruct
      (n_way_join_row_like_for_closures join_env_n_way_join_env_extension
         n_way_join_rec s function_slots)
      as [[joined s1] | ] eqn:Hjoin; [ | discriminate ].
    injection Hstep as <- <-.
    assert (Hargs_slots :
      args_in_scope
        (arg_ok free_variables_of_row_like_for_closures
           kinds_of_closures)
        s function_slots).
    { apply (args_of_parts _ _ _ _ s heads function_slots Hargs).
      intros index closures Hin.
      destruct (Hfrom index closures Hin)
        as [[-> ->] | (head & mode' & Hin' & Hread)].
      - exists (Closures by_function_slot alloc_mode).
        split; [ left; reflexivity | split; apply incl_refl ].
      - exists head. split; [ right; exact Hin' | ].
        rewrite (read_closures_eq head closures mode' Hread).
        split; apply incl_refl. }
    destruct
      (step_sound_at _ _ _ _ _ _ n_way_join_row_like_for_closures_is_sound s
         s s1 function_slots joined Hwf Hkinded (grows_refl s) Hvalid
         Hd_slots Hargs_slots Hjoin)
      as (Hvalid1 & Hgrows1 & Hk1 & Hholds).
    destruct (Hholds s1 (grows_refl s1)) as (Htarget & Hbound & Hkinds).
    split; [ exact Hvalid1 | ]. split; [ exact Hgrows1 | ].
    split; [ exact Htarget | ].
    split; [ | intros Hk; split; [ exact (Hk1 Hk) | exact (Hkinds Hk) ] ].
    intros index head use statics valuation valuation' Hin Huse Hsatisfied
      Hextends v Hv.
    assert (Hhead :
      ∃ (closures : row_like_for_closures) (mode_i : Alloc_mode.t),
        head = Closures closures mode_i ∧ In (index, closures) function_slots).
    { destruct Hin as [[= <- <-] | Hin].
      - exists by_function_slot, alloc_mode.
        split; [ reflexivity | exact Hfirst ].
      - destruct (Hto index head Hin) as (closures & mode_i & Hread & Hin').
        exists closures, mode_i.
        split; [ exact (read_closures_eq head closures mode_i Hread) | ].
        exact Hin'. }
    destruct Hhead as (closures & mode_i & -> & Hin_i).
    destruct Hv as (slot & functions & value_slots & Hclosure & Hrow).
    exists slot, functions, value_slots. split; [ exact Hclosure | ].
    exact
      (Hbound index closures use statics valuation valuation' Hin_i Huse
         Hsatisfied Hextends (v, slot, functions, value_slots) Hrow).
  - (* Strings. *)
    rewrite gather_heads_eq in Hstep.
    destruct (indexed_map_opt (fun _ head => read_string head) other_heads)
      as [l | ] eqn:Hl; cbn [option_map] in Hstep;
      [ | exact (Hunknown s Hvalid (grows_refl s) (fun H => H) Hstep) ].
    injection Hstep as <- <-.
    destruct (indexed_map_opt_spec _ _ l Hl) as (_ & Hto & _).
    split; [ exact Hvalid | ]. split; [ apply grows_refl | ].
    split; [ intros x [] | ].
    split; [ | intros Hk; split; [ exact Hk | ] ].
    2: { intros simple kind Hin. cbn [of_known] in Hin. simpl in Hin.
         refine (answer_check_passes s _ simple kind _ Hin). intros Hchecks.
         assert (Hfirst_ok :=
           arg_check_passes _ _ s heads first_id (String strs)
             (non_empty strs) Hargs (or_introl eq_refl) (incl_refl _)
             Hchecks).
         destruct strs as [ | bytes strs0]; [ discriminate Hfirst_ok | ].
         pose proof
           (fold_strings_keeps l (bytes :: strs0) bytes (or_introl eq_refl))
           as Hkeeps.
         destruct (List.fold_right _ _ l);
           [ destruct Hkeeps | reflexivity ]. }
    intros index head use statics valuation valuation' Hin _ _ _ v Hv.
    destruct Hin as [[= <- <-] | Hin].
    + destruct Hv as (bytes & Hstring & Hbytes). exists bytes.
      split; [ exact Hstring | apply fold_strings_keeps; exact Hbytes ].
    + destruct (Hto index head Hin) as (strs' & Hread & Hin').
      destruct head; try discriminate. injection Hread as ->.
      destruct Hv as (bytes & Hstring & Hbytes). exists bytes.
      split; [ exact Hstring | ].
      exact (fold_strings_in l strs index strs' bytes Hin' Hbytes).
Qed.

(* ------------------------------------------------------------------------ *)
(* Heads of kind value                                                      *)
(* ------------------------------------------------------------------------ *)

Definition non_null_of (head : head_of_kind_value)
    : or_unknown_or_bottom head_of_kind_value_non_null :=
  match head with
  | Head_of_kind_value non_null _ => non_null
  end.

Definition variables_of_non_null
    (non_null : or_unknown_or_bottom head_of_kind_value_non_null)
    : list variable :=
  match non_null with
  | Ok head => free_variables_of_head_of_kind_value_non_null head
  | Unknown | Bottom => []
  end.

Definition aliases_of_non_null
    (non_null : or_unknown_or_bottom head_of_kind_value_non_null)
    : list (simple * flambda_kind) :=
  match non_null with
  | Ok head => kinds_of_non_null head
  | Unknown | Bottom => []
  end.

(* What a head of kind value says of a value that is not null. *)
Definition non_null_part_holds (statics : static_env) (vars : env) (v : value)
    (non_null : or_unknown_or_bottom head_of_kind_value_non_null) : Prop :=
  match non_null with
  | Unknown => kind_of_value v = Value
  | Bottom => False
  | Ok head => head_of_kind_value_non_null_satisfied statics vars v head
  end.

Definition non_null_part_bound (statics : static_env) (valuation : env)
    (head : head_of_kind_value) (valuation' : env)
    (non_null : or_unknown_or_bottom head_of_kind_value_non_null) : Prop :=
  ∀ v : value,
    non_null_part_holds statics valuation v (non_null_of head) ->
    non_null_part_holds statics valuation' v non_null.

Lemma non_null_part_kind (statics : static_env) (vars : env) (v : value)
    (non_null : or_unknown_or_bottom head_of_kind_value_non_null) :
  non_null_part_holds statics vars v non_null -> kind_of_value v = Value.
Proof.
  destruct non_null as [ | head | ]; simpl;
    [ auto | intros Hhead | intros [] ].
  exact (proj1 (head_non_null_kind statics vars v head Hhead)).
Qed.

Lemma ok_head_in (heads : list (Index.t * head_of_kind_value))
    (index : Index.t) (head : head_of_kind_value_non_null) :
  In (index, head) (flat_map ok_head heads)
  <-> ∃ is_null : is_null,
        In (index, Head_of_kind_value (Ok head) is_null) heads.
Proof.
  rewrite in_flat_map. split.
  - intros ([id [non_null is_null]] & Hin & Hhead). unfold ok_head in Hhead.
    simpl in Hhead.
    destruct non_null as [ | head' | ]; [ destruct Hhead | | destruct Hhead ].
    destruct Hhead as [[= <- <-] | []]. exists is_null. exact Hin.
  - intros (is_null & Hin).
    exists (index, Head_of_kind_value (Ok head) is_null).
    split; [ exact Hin | left; reflexivity ].
Qed.

Lemma ok_head_distinct (heads : list (Index.t * head_of_kind_value)) :
  NoDup (map fst heads) -> NoDup (map fst (flat_map ok_head heads)).
Proof.
  apply flat_map_indexed_distinct.
  - intros [id [non_null is_null]] index head. unfold ok_head. simpl.
    destruct non_null as [ | head' | ];
      [ intros [] | intros [[= <- _] | []]; reflexivity | intros [] ].
  - intros [id [non_null is_null]]. unfold ok_head. simpl.
    destruct non_null as [ | head' | ]; try apply NoDup_nil.
    apply NoDup_cons; [ intros [] | apply NoDup_nil ].
Qed.

Local Abbreviation value_head_bound :=
  (known_bound
     (all_kept head_of_kind_value_satisfied head_of_kind_value_satisfied)).

Theorem n_way_join_head_of_kind_value_is_sound :
  step_sound free_variables_of_head_of_kind_value
    kinds_of_head
    (of_known free_variables_of_head_of_kind_value)
    (of_known kinds_of_head) value_head_bound
    (n_way_join_head_of_kind_value n_way_join_simples import_type
       join_env_n_way_join_env_extension type_eqb block_shape_eqb
       n_way_join_rec).
Proof.
  intros s s' heads answer Hvalid Hwf Hkinded Hdistinct Hargs Hstep.
  unfold n_way_join_head_of_kind_value in Hstep.
  destruct (n_way_join_is_null n_way_join_simples s heads)
    as [[is_null sa] | ] eqn:His_null; [ | discriminate ].
  assert (Hargs_null :
    args_in_scope
      (arg_ok free_variables_of_head_of_kind_value
         aliases_of_head_of_kind_value)
      s heads).
  { intros index head Hin.
    destruct (Hargs index head Hin) as (use & Huse & Hvariables & Hkinds).
    exists use. split; [ exact Huse | ]. split; [ exact Hvariables | ].
    exact
      (kinds_of_head_ask_aliases (simple_of_kind use) with_equations
         with_checks head (ill_formed_of_no_kind use) Hkinds). }
  destruct
    (step_sound_at _ _ _ _ _ _
       (n_way_join_is_null_is_sound n_way_join_simples Hsimples) s s sa heads
       is_null Hwf Hkinded (grows_refl s) Hvalid Hdistinct Hargs_null
       His_null)
    as (Hvalida & Hgrowsa & Hka & Hholds_is_null).
  (* What is left to do once the heads that are not null are joined. *)
  assert (Hfinal :
    ∀ (non_null : or_unknown_or_bottom head_of_kind_value_non_null)
      (sb : state),
      state_valid sb -> grows sa sb -> (state_kinded sa -> state_kinded sb) ->
      answer_holds_from variables_of_non_null aliases_of_non_null
        non_null_part_bound sa sb heads non_null ->
      match non_null, is_null with
      | Unknown, Maybe_null None => Some (Or_unknown.Unknown, sb)
      | _, _ =>
        Some (Or_unknown.Known (Head_of_kind_value non_null is_null), sb)
      end
      = Some (answer, s') ->
      state_valid s' ∧ grows s s'
      ∧ in_target (of_known free_variables_of_head_of_kind_value) s' answer
      ∧ (∀ (index : Index.t) (x : head_of_kind_value) (use : Typing_env.t)
           (statics : static_env) (valuation valuation' : env),
           In (index, x) heads ->
           joined_env s' index = Some use ->
           typing_env_satisfied_strictly statics valuation use ->
           extends s' index valuation valuation' ->
           value_head_bound statics valuation x valuation' answer)
      ∧ (state_kinded s ->
         state_kinded s'
         ∧ kinded_in_target (of_known kinds_of_head) s'
             answer)).
  { intros non_null sb Hvalidb Hgrowsb Hkb Hholds_non_null Hanswer.
    assert (Hcases :
      s' = sb
      ∧ (answer = Or_unknown.Unknown
         ∨ answer
           = Or_unknown.Known (Head_of_kind_value non_null is_null))).
    { destruct non_null as [ | head | ]; destruct is_null as [ | [y | ]];
        cbv beta iota in Hanswer;
        (split;
         [ congruence | first [ left; congruence | right; congruence ] ]). }
    destruct Hcases as (-> & Hanswer').
    split; [ exact Hvalidb | ].
    split; [ exact (grows_trans _ _ _ Hgrowsa Hgrowsb) | ].
    destruct Hanswer' as [-> | ->].
    { split; [ intros x [] | ]. split; [ intros; exact I | ].
      intros Hk. split; [ exact (Hkb (Hka Hk)) | intros simple kind [] ]. }
    destruct (Hholds_non_null sb (grows_refl sb))
      as (Htarget_non_null & Hbound_non_null & Hkinds_non_null).
    destruct (Hholds_is_null sb Hgrowsb)
      as (Htarget_is_null & Hbound_is_null & Hkinds_is_null).
    split; [ | split ].
    - intros x Hx. simpl in Hx. apply in_app_or in Hx as [Hx | Hx];
        [ exact (Htarget_non_null x Hx) | exact (Htarget_is_null x Hx) ].
    - intros index [non_null_i is_null_i] use statics valuation valuation'
        Hin Huse Hsatisfied Hextends v Hv.
      pose proof
        (Hbound_non_null index _ use statics valuation valuation' Hin Huse
           Hsatisfied Hextends)
        as Hnon_null_b.
      pose proof
        (Hbound_is_null index _ use statics valuation valuation' Hin Huse
           Hsatisfied Hextends)
        as His_null_b.
      cbn [is_null_of_head] in His_null_b.
      destruct Hv as [(-> & Hnull) | (Hnot_null & Hnon_null & Hnull)].
      + left. split; [ reflexivity | ].
        destruct is_null_i as [ | x]; [ destruct Hnull | ].
        pose proof (His_null_b true Hnull) as Hjoined.
        destruct is_null as [ | y]; [ discriminate Hjoined | exact Hjoined ].
      + right. split; [ exact Hnot_null | ].
        split; [ exact (Hnon_null_b v Hnon_null) | ].
        assert (Hjoined : is_null_says valuation' is_null false).
        { apply His_null_b. destruct is_null_i as [ | x];
            [ reflexivity | exact Hnull ]. }
        destruct is_null as [ | y]; [ exact I | exact Hjoined ].
    - intros Hk. split; [ exact (Hkb (Hka Hk)) | ].
      intros simple kind Hsimple. simpl in Hsimple.
      apply in_app_or in Hsimple as [Hsimple | Hsimple];
        [ exact (Hkinds_non_null (Hka Hk) simple kind Hsimple)
        | exact (Hkinds_is_null Hk simple kind Hsimple) ]. }
  (* The part that is not null is unknown. *)
  assert (Hunknown :
    ∀ sb : state,
      answer_holds_from variables_of_non_null aliases_of_non_null
        non_null_part_bound sa sb heads Unknown).
  { intros sb s'' _. split; [ intros x [] | ].
    split; [ | intros _ simple kind [] ].
    intros index head use statics valuation valuation' _ _ _ _ v Hv.
    exact (non_null_part_kind statics valuation v _ Hv). }
  destruct (non_null_heads [] heads) as [non_nulls | ] eqn:Hnon_nulls.
  2: { exact
         (Hfinal Unknown sa Hvalida (grows_refl sa) (fun H => H)
            (Hunknown sa) Hstep). }
  destruct (non_null_heads_spec heads [] non_nulls Hnon_nulls)
    as (Heq & Hknown).
  rewrite app_nil_r in Heq.
  (* A head whose part that is not null has a value is among those
     joined. *)
  assert (Hin_non_nulls :
    ∀ (index : Index.t)
      (non_null : or_unknown_or_bottom head_of_kind_value_non_null)
      (is_null_i : FlambdaTypeGrammar.is_null) (statics : static_env)
      (valuation : env) (v : value),
      In (index, Head_of_kind_value non_null is_null_i) heads ->
      non_null_part_holds statics valuation v non_null ->
      ∃ head : head_of_kind_value_non_null,
        non_null = Ok head ∧ In (index, head) non_nulls).
  { intros index non_null is_null_i statics valuation v Hin Hv.
    destruct non_null as [ | head | ];
      [ destruct (Hknown index _ _ Hin eq_refl) | | destruct Hv ].
    exists head. split; [ reflexivity | ]. rewrite Heq, <- in_rev.
    apply ok_head_in. exists is_null_i. exact Hin. }
  destruct non_nulls as [ | first rest].
  { apply (Hfinal Bottom sa Hvalida (grows_refl sa) (fun H => H));
      [ | exact Hstep ].
    intros s'' _. split; [ intros x [] | ].
    split; [ | intros _ simple kind [] ].
    intros index [non_null is_null_i] use statics valuation valuation' Hin _
      _ _ v Hv.
    destruct (Hin_non_nulls index non_null is_null_i statics valuation v Hin
                Hv)
      as (head & _ & []). }
  set (non_nulls := first :: rest) in *.
  assert (Hd_non_nulls : join_args_distinct non_nulls).
  { unfold join_args_distinct. rewrite Heq, map_rev. apply NoDup_rev.
    apply ok_head_distinct. exact Hdistinct. }
  assert (Hargs_non_nulls :
    args_in_scope
      (arg_ok free_variables_of_head_of_kind_value_non_null
         kinds_of_non_null)
      s non_nulls).
  { apply (args_of_parts _ _ _ _ s heads non_nulls Hargs).
    intros index head Hin. rewrite Heq, <- in_rev in Hin.
    apply ok_head_in in Hin as (is_null_i & Hin).
    exists (Head_of_kind_value (Ok head) is_null_i).
    split; [ exact Hin | ]. simpl. split; apply incl_appl, incl_refl. }
  destruct
    (n_way_join_head_of_kind_value_non_null n_way_join_simples import_type
       join_env_n_way_join_env_extension type_eqb block_shape_eqb
       n_way_join_rec sa non_nulls)
    as [[joined sb] | ] eqn:Hjoin; [ | discriminate ].
  destruct
    (step_sound_at _ _ _ _ _ _
       n_way_join_head_of_kind_value_non_null_is_sound s sa sb non_nulls
       joined Hwf Hkinded Hgrowsa Hvalida Hd_non_nulls Hargs_non_nulls Hjoin)
    as (Hvalidb & Hgrowsb & Hkb & Hholds).
  destruct joined as [head' | ]; cbv beta iota in Hstep.
  2: { exact (Hfinal Unknown sb Hvalidb Hgrowsb Hkb (Hunknown sb) Hstep). }
  apply (Hfinal (Ok head') sb Hvalidb Hgrowsb Hkb); [ | exact Hstep ].
  intros s'' Hgrows''.
  destruct (Hholds s'' Hgrows'') as (Htarget & Hbound & Hkinds).
  split; [ exact Htarget | ]. split; [ | exact Hkinds ].
  intros index [non_null is_null_i] use statics valuation valuation' Hin Huse
    Hsatisfied Hextends v Hv.
  destruct (Hin_non_nulls index non_null is_null_i statics valuation v Hin Hv)
    as (head & -> & Hin').
  exact
    (Hbound index head use statics valuation valuation' Hin' Huse Hsatisfied
       Hextends v Hv).
Qed.

(* ------------------------------------------------------------------------ *)
(* Expanded types                                                           *)
(* ------------------------------------------------------------------------ *)

Definition variables_of_expanded (e : Expanded_type.t) : list variable :=
  free_variables (Expanded_type.to_type e).

Definition aliases_of_expanded (e : Expanded_type.t)
    : list (simple * flambda_kind) :=
  kinds (Expanded_type.to_type e).

Definition expanded_holds (statics : static_env) (vars : env) (v : value)
    (e : Expanded_type.t) : Prop :=
  type_satisfied statics vars v (Expanded_type.to_type e).

(* The expanded types given are of one kind. *)
Definition expandeds_of_kind (kind : flambda_kind)
    (expandeds : list (Index.t * Expanded_type.t)) : Prop :=
  ∀ (index : Index.t) (e : Expanded_type.t),
    In (index, e) expandeds ->
    Expanded_type.kind e = kind
    ∧ FlambdaTypeGrammar.kind (Expanded_type.to_type e) = kind.

(* What is proved of the expanded type [e'] that
   [n_way_join_expanded_head] answers at [s2], given [expandeds] at
   [s1]. *)
Definition expanded_answer_sound (kind : flambda_kind) (s1 s2 : state)
    (expandeds : list (Index.t * Expanded_type.t)) (e' : Expanded_type.t)
    : Prop :=
  state_valid s2 ∧ grows s1 s2 ∧ (state_kinded s1 -> state_kinded s2)
  ∧ FlambdaTypeGrammar.kind (Expanded_type.to_type e') = kind
  ∧ answer_holds_from variables_of_expanded aliases_of_expanded
      (all_kept expanded_holds expanded_holds) s1 s2 expandeds e'.

Lemma expanded_unknown_sound (kind : flambda_kind) (s1 s2 : state)
    (expandeds : list (Index.t * Expanded_type.t)) :
  state_valid s2 -> grows s1 s2 -> (state_kinded s1 -> state_kinded s2) ->
  expandeds_of_kind kind expandeds ->
  expanded_answer_sound kind s1 s2 expandeds
    (Expanded_type.create_unknown kind).
Proof.
  intros Hvalid Hgrows Hk Hkinds.
  split; [ exact Hvalid | ]. split; [ exact Hgrows | ].
  split; [ exact Hk | ]. split; [ apply kind_unknown | ].
  intros s' _.
  split.
  { intros x Hx. change (x ∈ free_variables (unknown kind)) in Hx.
    rewrite free_variables_unknown in Hx. destruct Hx. }
  split.
  2: { intros _ simple kind' Hsimple.
       change ((simple, kind') ∈ kinds (unknown kind))
         in Hsimple.
       rewrite kinds_with_unknown in Hsimple. destruct Hsimple. }
  intros index e use statics valuation valuation' Hin _ _ _ v Hv.
  unfold expanded_holds in *.
  change (Expanded_type.to_type (Expanded_type.create_unknown kind))
    with (unknown kind).
  apply unknown_satisfied. rewrite (type_satisfied_kind _ _ _ _ Hv).
  exact (proj2 (Hkinds index e Hin)).
Qed.

Lemma expanded_bottom_sound (kind : flambda_kind) (s1 : state) :
  state_valid s1 ->
  expanded_answer_sound kind s1 s1 [] (Expanded_type.create_bottom kind).
Proof.
  intros Hvalid.
  split; [ exact Hvalid | ]. split; [ apply grows_refl | ].
  split; [ auto | ]. split; [ apply kind_bottom | ].
  intros s' _.
  split.
  { intros x Hx. change (x ∈ free_variables (bottom kind)) in Hx.
    rewrite free_variables_bottom in Hx. destruct Hx. }
  split; [ intros ? ? ? ? ? ? [] | ].
  intros _ simple kind' Hsimple.
  change ((simple, kind') ∈ kinds (bottom kind)) in Hsimple.
  rewrite kinds_with_bottom in Hsimple. destruct Hsimple.
Qed.

(* An expanded type that is bottom, before those joined. *)
Lemma expanded_answer_cons_bottom (kind : flambda_kind) (s1 s2 : state)
    (id : Index.t) (e : Expanded_type.t)
    (expandeds : list (Index.t * Expanded_type.t)) (e' : Expanded_type.t) :
  Expanded_type.descr e = Bottom ->
  expanded_answer_sound kind s1 s2 expandeds e' ->
  expanded_answer_sound kind s1 s2 ((id, e) :: expandeds) e'.
Proof.
  intros Hdescr (Hvalid & Hgrows & Hk & Hkind & Hholds).
  split; [ exact Hvalid | ]. split; [ exact Hgrows | ].
  split; [ exact Hk | ]. split; [ exact Hkind | ].
  intros s' Hgrows'.
  destruct (Hholds s' Hgrows') as (Htarget & Hbound & Hkinds).
  split; [ exact Htarget | ]. split; [ | exact Hkinds ].
  intros index e0 use statics valuation valuation' [[= <- <-] | Hin] Huse
    Hsatisfied Hextends v Hv.
  - unfold expanded_holds, Expanded_type.to_type in Hv. rewrite Hdescr in Hv.
    destruct (bottom_satisfied statics valuation v _ Hv).
  - exact
      (Hbound index e0 use statics valuation valuation' Hin Huse Hsatisfied
         Hextends v Hv).
Qed.

(* The join of the heads of one sort of the expanded types that are
   not bottom, the first of which has such a head: [extract] reads a
   head of that sort, [inject] makes an expanded type of one, and
   [type_of_head] is the type that it stands for. *)
Lemma expanded_heads_joined {H : Type}
    (extract : Expanded_type.head -> option H)
    (inject : H -> Expanded_type.t) (type_of_head : H -> flambda_type)
    (variables_of_head : H -> list variable)
    (aliases_of_head : H -> list (simple * flambda_kind))
    (variables_of_answer : Or_unknown.t H -> list variable)
    (aliases_of_answer : Or_unknown.t H -> list (simple * flambda_kind))
    (head_holds : static_env -> env -> value -> H -> Prop)
    (join : state -> list (Index.t * H) -> option (Or_unknown.t H * state))
    (kind : flambda_kind) (s s1 s2 : state) (id1 : Index.t)
    (expanded1 : Expanded_type.t) (head1 : Expanded_type.head) (a1 : H)
    (rest : list (Index.t * Expanded_type.t))
    (heads_opt : option (list (Index.t * H)))
    (joined : Or_unknown.t Expanded_type.t) :
  step_sound variables_of_head aliases_of_head variables_of_answer
    aliases_of_answer (known_bound (all_kept head_holds head_holds)) join ->
  (∀ (e : Expanded_type.t) (head : Expanded_type.head) (a : H),
     Expanded_type.descr e = Ok head ->
     extract head = Some a -> Expanded_type.to_type e = type_of_head a) ->
  (∀ a : H, Expanded_type.to_type (inject a) = type_of_head a) ->
  (∀ a : H, free_variables (type_of_head a) = variables_of_head a) ->
  (∀ a : H, kinds (type_of_head a) = aliases_of_head a) ->
  (∀ a : H, variables_of_answer (Or_unknown.Known a) = variables_of_head a) ->
  (∀ a : H, aliases_of_answer (Or_unknown.Known a) = aliases_of_head a) ->
  (∀ (statics : static_env) (vars : env) (v : value) (a : H),
     type_satisfied statics vars v (type_of_head a)
     <-> head_holds statics vars v a) ->
  (∀ a a' : H,
     FlambdaTypeGrammar.kind (type_of_head a)
     = FlambdaTypeGrammar.kind (type_of_head a')) ->
  envs_well_formed s -> envs_kinded s -> grows s s1 -> state_valid s1 ->
  join_args_distinct ((id1, expanded1) :: rest) ->
  args_in_scope (arg_ok variables_of_expanded aliases_of_expanded) s
    ((id1, expanded1) :: rest) ->
  expandeds_of_kind kind ((id1, expanded1) :: rest) ->
  Expanded_type.descr expanded1 = Ok head1 ->
  extract head1 = Some a1 ->
  extract_heads extract rest = Some heads_opt ->
  match heads_opt with
  | None => Some (Or_unknown.Unknown, s1)
  | Some heads => map_join_result inject (join s1 ((id1, a1) :: heads))
  end
  = Some (joined, s2) ->
  expanded_answer_sound kind s1 s2 ((id1, expanded1) :: rest)
    match joined with
    | Or_unknown.Known e => e
    | Or_unknown.Unknown => Expanded_type.unknown_like expanded1
    end.
Proof.
  intros Hjoin_sound Hextract_type Hinject Hvariables Haliases
    Hvariables_answer Haliases_answer Hholds_iff Hkind_head Hwf Hkinded
    Hgrows Hvalid Hdistinct Hargs Hkinds Hdescr1 Ha1 Hextract Hjoined.
  set (all := (id1, expanded1) :: rest) in *.
  assert (Hkind1 : Expanded_type.kind expanded1 = kind).
  { apply (Hkinds id1 expanded1). left. reflexivity. }
  assert (Hunknown :
    ∀ s2' : state,
      state_valid s2' -> grows s1 s2' ->
      (state_kinded s1 -> state_kinded s2') ->
      expanded_answer_sound kind s1 s2' all
        (Expanded_type.unknown_like expanded1)).
  { intros s2' Hvalid2 Hgrows2 Hk2. unfold Expanded_type.unknown_like.
    rewrite Hkind1.
    exact (expanded_unknown_sound kind s1 s2' all Hvalid2 Hgrows2 Hk2 Hkinds). }
  destruct heads_opt as [heads | ].
  2: { injection Hjoined as <- <-.
       exact (Hunknown s1 Hvalid (grows_refl s1) (fun H => H)). }
  destruct (extract_heads_spec extract rest heads Hextract) as (-> & Hall).
  assert (Hfirst : extracted extract (id1, expanded1) = [(id1, a1)]).
  { unfold extracted. cbn [fst snd]. rewrite Hdescr1, Ha1. reflexivity. }
  assert (Heq :
    (id1, a1) :: flat_map (extracted extract) rest
    = flat_map (extracted extract) all).
  { unfold all. cbn [flat_map]. rewrite Hfirst. reflexivity. }
  rewrite Heq in Hjoined. clear Heq.
  set (heads := flat_map (extracted extract) all) in *.
  (* Each expanded type given is bottom or has a head that was read. *)
  assert (Hall' :
    ∀ (index : Index.t) (e : Expanded_type.t),
      In (index, e) all ->
      Expanded_type.descr e = Bottom
      ∨ ∃ a : H,
          Expanded_type.to_type e = type_of_head a ∧ In (index, a) heads).
  { intros index e Hin.
    assert (Hcases :
      Expanded_type.descr e = Bottom
      ∨ ∃ (head : Expanded_type.head) (a : H),
          Expanded_type.descr e = Ok head ∧ extract head = Some a).
    { destruct Hin as [[= <- <-] | Hin];
        [ right; exists head1, a1; split; assumption | exact (Hall _ _ Hin) ]. }
    destruct Hcases as [Hbottom | (head & a & Hdescr & Ha)];
      [ left; exact Hbottom | right ].
    exists a. split; [ exact (Hextract_type e head a Hdescr Ha) | ].
    apply extracted_in. exists e, head. auto. }
  unfold map_join_result in Hjoined.
  destruct (join s1 heads) as [[joined_head s2'] | ] eqn:Hjoin;
    [ | discriminate ].
  injection Hjoined as <- <-.
  assert (Hd_heads : join_args_distinct heads).
  { apply extracted_distinct. exact Hdistinct. }
  assert (Hargs_heads :
    args_in_scope (arg_ok variables_of_head aliases_of_head) s heads).
  { apply (args_of_parts _ _ _ _ s all heads Hargs). intros index a Hin.
    apply extracted_in in Hin as (e & head & Hin & Hdescr & Ha).
    exists e. split; [ exact Hin | ].
    unfold variables_of_expanded, aliases_of_expanded.
    rewrite (Hextract_type e head a Hdescr Ha), Hvariables, Haliases.
    split; apply incl_refl. }
  destruct
    (step_sound_at _ _ _ _ _ _ Hjoin_sound s s1 s2' heads joined_head Hwf
       Hkinded Hgrows Hvalid Hd_heads Hargs_heads Hjoin)
    as (Hvalid2 & Hgrows2 & Hk2 & Hholds).
  destruct joined_head as [a' | ]; cbn [or_unknown_map];
    [ | exact (Hunknown s2' Hvalid2 Hgrows2 Hk2) ].
  split; [ exact Hvalid2 | ]. split; [ exact Hgrows2 | ].
  split; [ exact Hk2 | ].
  split.
  { rewrite Hinject, (Hkind_head a' a1),
      <- (Hextract_type expanded1 head1 a1 Hdescr1 Ha1).
    apply (Hkinds id1 expanded1). left. reflexivity. }
  intros s' Hgrows'.
  destruct (Hholds s' Hgrows') as (Htarget & Hbound & Hkinds').
  split.
  { intros x Hx. unfold variables_of_expanded in Hx.
    rewrite Hinject, Hvariables, <- Hvariables_answer in Hx.
    exact (Htarget x Hx). }
  split.
  2: { intros Hk simple kind' Hsimple. unfold aliases_of_expanded in Hsimple.
       rewrite Hinject, Haliases, <- Haliases_answer in Hsimple.
       exact (Hkinds' Hk simple kind' Hsimple). }
  intros index e use statics valuation valuation' Hin Huse Hsatisfied
    Hextends v Hv.
  unfold expanded_holds in *. rewrite Hinject.
  destruct (Hall' index e Hin) as [Hbottom | (a & Htype & Hin')].
  - unfold Expanded_type.to_type in Hv. rewrite Hbottom in Hv.
    destruct (bottom_satisfied statics valuation v _ Hv).
  - rewrite Htype in Hv. apply Hholds_iff. apply Hholds_iff in Hv.
    exact
      (Hbound index a use statics valuation valuation' Hin' Huse Hsatisfied
         Hextends v Hv).
Qed.

(* A step that asks nothing of the aliases, with a check on what it is
   given and one on what it answers: the check of the answer passes if
   those of the things given do. *)
Lemma step_sound_checked {X H : Type} (variables_of : X -> list variable)
    (variables_of_answer : Or_unknown.t H -> list variable)
    (test : X -> bool) (test_answer : H -> bool)
    (bound : static_env -> env -> X -> env -> Or_unknown.t H -> Prop)
    (step : state -> list (Index.t * X) -> option (Or_unknown.t H * state)) :
  step_sound variables_of (fun _ => []) variables_of_answer (fun _ => [])
    bound step ->
  (∀ (s s' : state) (args : list (Index.t * X)) (h : H),
     step s args = Some (Or_unknown.Known h, s') ->
     (∀ (index : Index.t) (x : X), In (index, x) args -> test x = true) ->
     test_answer h = true) ->
  step_sound variables_of (fun x => checked (test x) ++ [])
    variables_of_answer (of_known (fun h => checked (test_answer h) ++ []))
    bound step.
Proof.
  intros Hsound Htest s s' args answer Hvalid Hwf Hkinded Hdistinct Hargs
    Hstep.
  assert (Hargs' : args_in_scope (arg_ok variables_of (fun _ => [])) s args).
  { apply (args_of_parts _ _ _ _ s args args Hargs). intros index x Hin.
    exists x. split; [ exact Hin | ].
    split; [ apply incl_refl | intros p [] ]. }
  destruct (Hsound s s' args answer Hvalid Hwf Hkinded Hdistinct Hargs' Hstep)
    as (Hvalid' & Hgrows & Htarget & Hbound & Hkinds).
  split; [ exact Hvalid' | ]. split; [ exact Hgrows | ].
  split; [ exact Htarget | ]. split; [ exact Hbound | ].
  intros Hk. split; [ exact (proj1 (Hkinds Hk)) | ].
  destruct answer as [h | ]; [ | intros simple kind [] ].
  intros simple kind Hin. cbn [of_known] in Hin. rewrite app_nil_r in Hin.
  refine (answer_check_passes s' _ simple kind _ Hin). intros Hchecks.
  apply (Htest s s' args h Hstep). intros index x Hin'.
  exact
    (arg_check_passes _ _ s args index x (test x) Hargs Hin'
       (incl_appl _ (incl_refl _)) Hchecks).
Qed.

(* The head of naked immediates that is answered has a number. *)
Lemma naked_immediate_join_well_formed (s s' : state)
    (heads : list (Index.t * head_of_kind_naked_immediate))
    (h : head_of_kind_naked_immediate) :
  n_way_join_head_of_kind_naked_immediate joined_env
    reduce_head_of_kind_naked_immediate s heads
  = Some (Or_unknown.Known h, s') ->
  (∀ (index : Index.t) (x : head_of_kind_naked_immediate),
     In (index, x) heads ->
     head_of_kind_naked_immediate_well_formed x = true) ->
  head_of_kind_naked_immediate_well_formed h = true.
Proof.
  unfold n_way_join_head_of_kind_naked_immediate. intros Hstep _.
  destruct (join_naked_immediates _ _ _ _ _) as [[imms | ] | ];
    [ | discriminate | discriminate ].
  cbv beta iota in Hstep.
  destruct imms as [ | n imms]; [ discriminate | ].
  injection Hstep as <- _. reflexivity.
Qed.

(* The set of naked numbers that is answered has a number, if the
   first set given has. *)
Lemma naked_numbers_join_well_formed (k : naked_number_kind) (s s' : state)
    (heads : list (Index.t * list Z)) (h : list Z) :
  match heads with
  | [] => None
  | (id1, head1) :: heads =>
    n_way_join_head_of_kind_naked_numbers s (head1, id1) heads
  end
  = Some (Or_unknown.Known h, s') ->
  (∀ (index : Index.t) (x : list Z),
     In (index, x) heads -> has_naked_number_types k && non_empty x = true) ->
  has_naked_number_types k && non_empty h = true.
Proof.
  intros Hstep Hall. destruct heads as [ | [id1 head1] heads];
    [ discriminate | ].
  specialize (Hall id1 head1 (or_introl eq_refl)).
  apply andb_true_iff in Hall as [Hk Hne]. rewrite Hk. cbn [andb].
  injection Hstep as <- _. cbn [fst].
  destruct head1 as [ | n head1]; [ discriminate | ].
  assert (Hin :
    ∀ acc : list Z,
      n ∈ acc ->
      n ∈ List.fold_left (fun t1 t2 => numbers_union t1 (snd t2)) heads acc).
  { induction heads as [ | t heads IH]; intros acc Hacc; simpl;
      [ exact Hacc | ].
    apply IH. apply numbers_union_in. left. exact Hacc. }
  specialize (Hin (n :: head1) (or_introl eq_refl)).
  destruct (List.fold_left _ heads (n :: head1));
    [ destruct Hin | reflexivity ].
Qed.

(* A join of heads that answers a head of the first, whatever the
   heads: the heads of that sort say one thing. *)
Lemma constant_head_join_sound {H : Type}
    (holds : static_env -> env -> value -> H -> Prop)
    (f :
       state -> H * Index.t -> list (Index.t * H)
       -> option (Or_unknown.t H * state)) :
  (∀ (s : state) (t1 : H * Index.t) (ts : list (Index.t * H)),
     ∃ a : H, f s t1 ts = Some (Or_unknown.Known a, s)) ->
  (∀ (statics : static_env) (vars vars' : env) (v : value) (a a' : H),
     holds statics vars v a -> holds statics vars' v a') ->
  step_sound (fun _ : H => []) (fun _ => []) (fun _ : Or_unknown.t H => [])
    (fun _ => []) (known_bound (all_kept holds holds))
    (fun s heads =>
       match heads with
       | [] => None
       | (id1, head1) :: heads => f s (head1, id1) heads
       end).
Proof.
  intros Hf Hholds s s' args answer Hvalid Hwf Hkinded Hdistinct Hargs Hstep.
  destruct args as [ | [id1 head1] heads]; [ discriminate | ].
  destruct (Hf s (head1, id1) heads) as [a Ha]. rewrite Ha in Hstep.
  injection Hstep as <- <-.
  split; [ exact Hvalid | ]. split; [ apply grows_refl | ].
  split; [ intros x [] | ].
  split.
  - intros index a_i use statics valuation valuation' _ _ _ _ v Hv.
    exact (Hholds statics valuation valuation' v a_i a Hv).
  - intros Hk. split; [ exact Hk | intros simple kind [] ].
Qed.

Local Ltac expanded_case Hstep Hcase rest heads_opt Hextract joined s2'
    Hjoined :=
  match type of Hstep with
  | context [extract_heads ?extract rest] =>
    destruct (extract_heads extract rest) as [heads_opt | ] eqn:Hextract;
      [ | discriminate ]
  end;
  match type of Hstep with
  | (match ?m with Some _ => _ | None => _ end) = _ =>
    destruct m as [[joined s2'] | ] eqn:Hjoined; [ | discriminate ]
  end;
  refine (Hcase joined s2' _ Hstep).

Lemma n_way_join_expanded_head_is_sound (kind : flambda_kind) (s s1 : state) :
  envs_well_formed s -> envs_kinded s -> grows s s1 -> state_valid s1 ->
  ∀ (expandeds : list (Index.t * Expanded_type.t)) (s2 : state)
    (answer : Or_unknown.t Expanded_type.t),
    join_args_distinct expandeds ->
    args_in_scope (arg_ok variables_of_expanded aliases_of_expanded) s
      expandeds ->
    expandeds_of_kind kind expandeds ->
    n_way_join_expanded_head joined_env n_way_join_simples import_type
      join_env_n_way_join_env_extension reduce_head_of_kind_naked_immediate
      type_eqb block_shape_eqb n_way_join_rec s1 kind expandeds
    = Some (answer, s2) ->
    ∃ e' : Expanded_type.t,
      answer = Or_unknown.Known e'
      ∧ expanded_answer_sound kind s1 s2 expandeds e'.
Proof.
  intros Hwf Hkinded Hgrows Hvalid.
  induction expandeds as [ | [id1 expanded1] rest IH];
    intros s2 answer Hdistinct Hargs Hkinds Hstep.
  { injection Hstep as <- <-. exists (Expanded_type.create_bottom kind).
    split; [ reflexivity | apply expanded_bottom_sound; exact Hvalid ]. }
  cbn [n_way_join_expanded_head] in Hstep.
  destruct (Expanded_type.descr expanded1) as [ | descr1 | ] eqn:Hdescr1.
  - (* The first is unknown. *)
    injection Hstep as <- <-. exists (Expanded_type.create_unknown kind).
    split; [ reflexivity | ].
    exact
      (expanded_unknown_sound kind s1 s1 _ Hvalid (grows_refl s1)
         (fun H => H) Hkinds).
  - (* The first has a head. *)
    assert (Hcase :
      ∀ (joined : Or_unknown.t Expanded_type.t) (s2' : state),
        expanded_answer_sound kind s1 s2' ((id1, expanded1) :: rest)
          match joined with
          | Or_unknown.Known e => e
          | Or_unknown.Unknown => Expanded_type.unknown_like expanded1
          end ->
        match joined with
        | Or_unknown.Known expanded => Some (Or_unknown.Known expanded, s2')
        | Or_unknown.Unknown =>
          Some (Or_unknown.Known (Expanded_type.unknown_like expanded1), s2')
        end
        = Some (answer, s2) ->
        ∃ e' : Expanded_type.t,
          answer = Or_unknown.Known e'
          ∧ expanded_answer_sound kind s1 s2 ((id1, expanded1) :: rest) e').
    { intros joined s2' Hsound Hanswer.
      destruct joined as [e | ]; injection Hanswer as <- <-; eexists;
        (split; [ reflexivity | exact Hsound ]). }
    destruct descr1 as [head1 | head1 | k head1 | head1 | head1].
    + expanded_case Hstep Hcase rest heads_opt Hextract joined s2' Hjoined.
      refine
        (expanded_heads_joined
           (fun head =>
              match head with
              | Expanded_type.Head_value head => Some head
              | _ => None
              end)
           Expanded_type.create_value
           (fun head => Type_value (Ok (No_alias head)))
           free_variables_of_head_of_kind_value kinds_of_head
           (of_known free_variables_of_head_of_kind_value)
           (of_known kinds_of_head)
           head_of_kind_value_satisfied
           (n_way_join_head_of_kind_value n_way_join_simples import_type
              join_env_n_way_join_env_extension type_eqb block_shape_eqb
              n_way_join_rec)
           kind s s1 s2' id1 expanded1 _ head1 rest heads_opt joined
           n_way_join_head_of_kind_value_is_sound _ _ _ _ _ _ _ _ Hwf Hkinded
           Hgrows Hvalid Hdistinct Hargs Hkinds Hdescr1 eq_refl Hextract
           Hjoined);
        try (intros; reflexivity).
      intros e head a Hdescr Ha. unfold Expanded_type.to_type.
      rewrite Hdescr. destruct head; try discriminate. injection Ha as ->.
      reflexivity.
    + expanded_case Hstep Hcase rest heads_opt Hextract joined s2' Hjoined.
      refine
        (expanded_heads_joined
           (fun head =>
              match head with
              | Expanded_type.Head_naked_immediate head => Some head
              | _ => None
              end)
           Expanded_type.create_naked_immediate
           (fun head => Type_naked_immediate (Ok (No_alias head)))
           free_variables_of_naked_immediate
           (fun head =>
              checked (head_of_kind_naked_immediate_well_formed head) ++ [])
           (of_known free_variables_of_naked_immediate)
           (of_known
              (fun head =>
                 checked (head_of_kind_naked_immediate_well_formed head)
                 ++ []))
           head_of_kind_naked_immediate_satisfied
           (n_way_join_head_of_kind_naked_immediate joined_env
              reduce_head_of_kind_naked_immediate)
           kind s s1 s2' id1 expanded1 _ head1 rest heads_opt joined
           (step_sound_checked _ _ _ _ _ _
              (n_way_join_head_of_kind_naked_immediate_is_sound
                 reduce_head_of_kind_naked_immediate Hreduce)
              naked_immediate_join_well_formed)
           _ _ _ _ _ _ _ _ Hwf Hkinded Hgrows Hvalid Hdistinct Hargs Hkinds
           Hdescr1 eq_refl Hextract Hjoined);
        try (intros; reflexivity).
      intros e head a Hdescr Ha. unfold Expanded_type.to_type.
      rewrite Hdescr. destruct head; try discriminate. injection Ha as ->.
      reflexivity.
    + expanded_case Hstep Hcase rest heads_opt Hextract joined s2' Hjoined.
      refine
        (expanded_heads_joined
           (fun head =>
              match head with
              | Expanded_type.Head_naked_number k' head =>
                if kind_eqb (Naked_number k) (Naked_number k')
                then Some head
                else None
              | _ => None
              end)
           (Expanded_type.create_naked_number k)
           (fun head => Type_naked_number k (Ok (No_alias head)))
           (fun _ => [])
           (fun head =>
              checked (has_naked_number_types k && non_empty head) ++ [])
           (fun _ => [])
           (of_known
              (fun head =>
                 checked (has_naked_number_types k && non_empty head) ++ []))
           (fun _ _ v numbers => naked_numbers_satisfied k v numbers)
           (fun s heads =>
              match heads with
              | [] => None
              | (id1, head1) :: heads =>
                n_way_join_head_of_kind_naked_numbers s (head1, id1) heads
              end)
           kind s s1 s2' id1 expanded1 _ head1 rest heads_opt joined
           (step_sound_checked _ _ _ _ _ _
              (n_way_join_head_of_kind_naked_numbers_is_sound k)
              (naked_numbers_join_well_formed k))
           _ _ _ _ _ _ _ _
           Hwf Hkinded Hgrows Hvalid Hdistinct Hargs Hkinds Hdescr1 _ Hextract
           Hjoined);
        try (intros; reflexivity).
      * intros e head a Hdescr Ha. unfold Expanded_type.to_type.
        rewrite Hdescr. destruct head as [ | | k' numbers | | ];
          try discriminate.
        destruct (kind_eqb (Naked_number k) (Naked_number k')) eqn:Hkind;
          [ | discriminate ].
        injection Ha as ->. apply kind_eqb_true in Hkind.
        injection Hkind as <-. reflexivity.
      * cbv beta iota. rewrite kind_eqb_refl. reflexivity.
    + expanded_case Hstep Hcase rest heads_opt Hextract joined s2' Hjoined.
      refine
        (expanded_heads_joined
           (fun head =>
              match head with
              | Expanded_type.Head_rec_info head => Some head
              | _ => None
              end)
           Expanded_type.create_rec_info
           (fun head => Type_rec_info (Ok (No_alias head))) (fun _ => [])
           (fun _ => []) (fun _ => []) (fun _ => [])
           (fun _ _ v _ => kind_of_value v = Rec_info)
           (fun s heads =>
              match heads with
              | [] => None
              | (id1, head1) :: heads =>
                n_way_join_head_of_kind_rec_info s (head1, id1) heads
              end)
           kind s s1 s2' id1 expanded1 _ head1 rest heads_opt joined
           (constant_head_join_sound _ _ _ _) _ _ _ _ _ _ _ _ Hwf Hkinded
           Hgrows Hvalid Hdistinct Hargs Hkinds Hdescr1 eq_refl Hextract
           Hjoined);
        try (intros; reflexivity).
      * intros s0 t1 ts. exists (fst t1). reflexivity.
      * intros statics vars vars' v a a' Hv. exact Hv.
      * intros e head a Hdescr Ha. unfold Expanded_type.to_type.
        rewrite Hdescr. destruct head; try discriminate. injection Ha as ->.
        reflexivity.
    + expanded_case Hstep Hcase rest heads_opt Hextract joined s2' Hjoined.
      refine
        (expanded_heads_joined
           (fun head =>
              match head with
              | Expanded_type.Head_region head => Some head
              | _ => None
              end)
           Expanded_type.create_region
           (fun head => Type_region (Ok (No_alias head))) (fun _ => [])
           (fun _ => []) (fun _ => []) (fun _ => [])
           (fun _ _ v _ => kind_of_value v = Region)
           (fun s heads =>
              match heads with
              | [] => None
              | (id1, head1) :: heads =>
                n_way_join_head_of_kind_region s (head1, id1) heads
              end)
           kind s s1 s2' id1 expanded1 _ head1 rest heads_opt joined
           (constant_head_join_sound _ _ _ _) _ _ _ _ _ _ _ _ Hwf Hkinded
           Hgrows Hvalid Hdistinct Hargs Hkinds Hdescr1 eq_refl Hextract
           Hjoined);
        try (intros; reflexivity).
      * intros s0 t1 ts. exists tt. reflexivity.
      * intros statics vars vars' v a a' Hv. exact Hv.
      * intros e head a Hdescr Ha. unfold Expanded_type.to_type.
        rewrite Hdescr. destruct head; try discriminate. injection Ha as ->.
        reflexivity.
  - (* The first is bottom. *)
    assert (Hdistinct' : join_args_distinct rest).
    { inversion Hdistinct. assumption. }
    assert (Hargs' :
      args_in_scope (arg_ok variables_of_expanded aliases_of_expanded) s rest).
    { intros index e Hin. apply Hargs. right. exact Hin. }
    assert (Hkinds' : expandeds_of_kind kind rest).
    { intros index e Hin. exact (Hkinds index e (or_intror Hin)). }
    destruct (IH s2 answer Hdistinct' Hargs' Hkinds' Hstep)
      as (e' & -> & Hsound).
    exists e'. split; [ reflexivity | ].
    exact
      (expanded_answer_cons_bottom kind s1 s2 id1 expanded1 rest e' Hdescr1
         Hsound).
Qed.

(* ------------------------------------------------------------------------ *)
(* The join of types                                                        *)
(* ------------------------------------------------------------------------ *)

Lemma expand_heads_eq (s : state) (ts : list (Index.t * flambda_type)) :
  expand_heads joined_env expand_head s ts
  = indexed_map_opt
      (fun id ty =>
         match joined_env s id with
         | Some tenv => expand_head tenv ty
         | None => None
         end)
      ts.
Proof.
  induction ts as [ | [id ty] ts IH]; simpl; [ reflexivity | ]. rewrite IH.
  destruct (joined_env s id) as [tenv | ]; [ | reflexivity ].
  destruct (expand_head tenv ty); [ | reflexivity ].
  destruct (indexed_map_opt _ ts); reflexivity.
Qed.

(* The variables and the aliases of a type that is an alias. *)
Lemma get_alias_opt_parts (ty : flambda_type) (simple : simple) :
  get_alias_opt ty = Some simple ->
  free_variables ty = free_variables_of_simple simple
  ∧ (simple, kind ty) ∈ kinds ty.
Proof.
  destruct ty
    as [[ | [head | s] | ] | [ | [head | s] | ] | k [ | [head | s] | ]
       | [ | [head | s] | ] | [ | [head | s] | ]];
    try discriminate; intros [= ->]; (split; [ reflexivity | ]); simpl;
    try (apply in_or_app; right); left; reflexivity.
Qed.

(* The body of the join of types meets the contract of the join of
   types, if the join of types that it calls does. *)
Theorem n_way_join_body_is_sound :
  type_join_sound
    (n_way_join_body joined_env n_way_join_simples import_type
       join_env_n_way_join_env_extension expand_head
       reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb
       n_way_join_rec).
Proof.
  intros s s' ts result Hvalid Hwf Hkinded Hdistinct Hscope Hkinds Hstep.
  unfold n_way_join_body in Hstep.
  destruct ts as [ | [id1 t1] others]; [ discriminate | ].
  cbv zeta in Hstep.
  destruct
    (forallb (fun t => kind_eqb (kind t1) (kind (snd t))) others) eqn:Hall;
    cbn [negb] in Hstep; [ | discriminate ].
  set (ts := (id1, t1) :: others) in *.
  assert (Hkind_ts :
    ∀ (index : Index.t) (t : flambda_type),
      In (index, t) ts -> kind t = kind t1).
  { intros index t [[= <- <-] | Hin]; [ reflexivity | ].
    rewrite forallb_forall in Hall. symmetry.
    exact (kind_eqb_true _ _ (Hall (index, t) Hin)). }
  set (kept := filter (fun t => negb (is_obviously_bottom (snd t))) ts) in *.
  assert (Hkept :
    ∀ (index : Index.t) (t : flambda_type),
      In (index, t) kept
      <-> In (index, t) ts ∧ is_obviously_bottom t = false).
  { intros index t. unfold kept. rewrite filter_In. cbn [snd].
    rewrite negb_true_iff. reflexivity. }
  assert (Hkept_distinct : join_args_distinct kept).
  { apply filter_indexed_distinct. exact Hdistinct. }
  (* A type that has a value is not left out. *)
  assert (Hlive :
    ∀ (index : Index.t) (t : flambda_type) (statics : static_env)
      (valuation : env) (v : value),
      In (index, t) ts ->
      type_satisfied statics valuation v t -> In (index, t) kept).
  { intros index t statics valuation v Hin Hv. apply Hkept.
    split; [ exact Hin | ].
    destruct (is_obviously_bottom t) eqn:Hbottom; [ | reflexivity ].
    destruct (is_obviously_bottom_satisfied statics valuation v t Hbottom Hv). }
  rewrite alias_simples_eq in Hstep.
  destruct (indexed_map_opt (fun _ ty => get_alias_opt ty) kept)
    as [simples | ] eqn:Hsimples'.
  - (* Every type is an alias. *)
    destruct (indexed_map_opt_spec _ _ simples Hsimples')
      as (Hfst & Hto & Hfrom).
    destruct (n_way_join_simples s (kind t1) simples) as [[joined s1] | ]
      eqn:Hjoin; [ | discriminate ].
    assert (Hd_simples : join_args_distinct simples).
    { unfold join_args_distinct. rewrite Hfst. exact Hkept_distinct. }
    assert (Hk_simples :
      args_in_scope (fun use simple => simple_of_kind use simple (kind t1)) s
        simples).
    { intros index simple Hin.
      destruct (Hfrom index simple Hin) as (ty & Halias & Hin').
      apply Hkept in Hin' as [Hin' _].
      destruct (Hkinds index ty Hin') as (use & Huse & Hkinded_ty).
      exists use. split; [ exact Huse | ].
      rewrite <- (Hkind_ts index ty Hin'). apply Hkinded_ty.
      exact (proj2 (get_alias_opt_parts ty simple Halias)). }
    destruct
      (Hsimples s s1 _ simples joined Hvalid Hwf Hkinded Hd_simples
         Hk_simples Hjoin)
      as (Hvalid1 & Hgrows & Hbottom & _ & Hok & Hkinded1).
    assert (Hs' : s' = s1) by (destruct joined; congruence). subst s'.
    split; [ exact Hvalid1 | ]. split; [ exact Hgrows | ].
    destruct joined as [simple | ].
    + assert (Hresult :
        result = Or_unknown.Known (alias_type_of (kind t1) simple))
        by congruence.
      subst result.
      destruct (Hok simple eq_refl) as (Htarget & Hbound).
      split.
      * intros ty [= <-]. split; [ | split ].
        -- intros x Hx. rewrite free_variables_alias in Hx.
           exact (Htarget x Hx).
        -- intros index t_index Hin. rewrite kind_alias. symmetry.
           exact (Hkind_ts index t_index Hin).
        -- intros index t_index use statics valuation valuation' v Hin Huse
             Hsatisfied Hextends Hv.
           pose proof (Hlive index t_index statics valuation v Hin Hv)
             as Hin'.
           destruct (Hto index t_index Hin') as (simple_i & Halias & Hin_i).
           apply (get_alias_satisfied statics valuation v t_index simple_i
                    Halias)
             in Hv as [Heval Hkind_v].
           apply alias_satisfied. split.
           ++ exact
                (Hbound index simple_i use statics valuation valuation' v
                   Hin_i Huse Hsatisfied Hextends Heval).
           ++ rewrite Hkind_v. exact (Hkind_ts index t_index Hin).
      * intros Hk. destruct (Hkinded1 Hk) as (Hk1 & Hsimple).
        split; [ exact Hk1 | ].
        intros ty [= <-] simple' kind' Hin.
        rewrite kinds_with_alias in Hin.
        destruct Hin as [[= <- <-] | []]. exact (Hsimple simple eq_refl).
    + assert (Hresult : result = Or_unknown.Known (bottom (kind t1)))
        by congruence.
      subst result.
      assert (Hnone : kept = []).
      { pose proof (Hbottom eq_refl) as Hnil. subst simples.
        symmetry in Hfst. exact (map_eq_nil _ _ Hfst). }
      split.
      * intros ty [= <-]. split; [ | split ].
        -- intros x Hx. rewrite free_variables_bottom in Hx. destruct Hx.
        -- intros index t_index Hin. rewrite kind_bottom. symmetry.
           exact (Hkind_ts index t_index Hin).
        -- intros index t_index use statics valuation valuation' v Hin _ _ _
             Hv.
           pose proof (Hlive index t_index statics valuation v Hin Hv)
             as Hin'.
           rewrite Hnone in Hin'. destruct Hin'.
      * intros Hk. split; [ exact (proj1 (Hkinded1 Hk)) | ].
        intros ty [= <-] simple' kind' Hin.
        rewrite kinds_with_bottom in Hin. destruct Hin.
  - (* A type is not an alias: each is expanded. *)
    rewrite expand_heads_eq in Hstep.
    match type of Hstep with
    | context [indexed_map_opt ?f kept] =>
      destruct (indexed_map_opt f kept) as [expandeds | ] eqn:Hexpandeds;
        [ | discriminate ]
    end.
    destruct (indexed_map_opt_spec _ _ expandeds Hexpandeds)
      as (Hfst & Hto & Hfrom).
    unfold map_join_result in Hstep.
    destruct
      (n_way_join_expanded_head joined_env n_way_join_simples import_type
         join_env_n_way_join_env_extension
         reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb
         n_way_join_rec s (kind t1) expandeds)
      as [[answer s1] | ] eqn:Hjoin; [ | discriminate ].
    injection Hstep as <- <-.
    assert (Hexpanded :
      ∀ (index : Index.t) (e : Expanded_type.t),
        In (index, e) expandeds ->
        ∃ (ty : flambda_type) (use : Typing_env.t),
          In (index, ty) ts ∧ joined_env s index = Some use
          ∧ expand_head use ty = Some e).
    { intros index e Hin. destruct (Hfrom index e Hin) as (ty & He & Hin').
      apply Hkept in Hin' as [Hin' _].
      destruct (joined_env s index) as [use | ] eqn:Huse; [ | discriminate ].
      exists ty, use. auto. }
    assert (Hd_expandeds : join_args_distinct expandeds).
    { unfold join_args_distinct. rewrite Hfst. exact Hkept_distinct. }
    assert (Hargs_expandeds :
      args_in_scope (arg_ok variables_of_expanded aliases_of_expanded) s
        expandeds).
    { intros index e Hin.
      destruct (Hexpanded index e Hin) as (ty & use & Hin' & Huse & He).
      exists use. split; [ exact Huse | ].
      destruct (Hexpand use ty e He) as (_ & Hscope_e & _).
      destruct (Hscope index ty Hin') as (use' & Huse' & Hscope_ty).
      destruct (Hkinds index ty Hin') as (use'' & Huse'' & Hkinded_ty).
      rewrite Huse in Huse', Huse''. injection Huse' as <-.
      injection Huse'' as <-.
      split;
        [ exact (Hscope_e (Hwf index use Huse) Hscope_ty)
        | exact
            (Hexpand_kinds use ty e He (Hwf index use Huse)
               (Hkinded index use Huse) Hkinded_ty) ]. }
    assert (Hkinds_expandeds : expandeds_of_kind (kind t1) expandeds).
    { intros index e Hin.
      destruct (Hexpanded index e Hin) as (ty & use & Hin' & Huse & He).
      destruct (Hexpand use ty e He) as (_ & _ & _ & Hkind_e).
      destruct (Hkinds index ty Hin') as (use' & Huse' & Hkinded_ty).
      rewrite Huse in Huse'. injection Huse' as <-.
      rewrite <- (Hkind_ts index ty Hin').
      exact
        (Hkind_e (Hconstants s index use Hvalid Hwf Hkinded Huse)
           (kinded_with_aliases use ty Hkinded_ty)). }
    destruct
      (n_way_join_expanded_head_is_sound (kind t1) s s Hwf Hkinded
         (grows_refl s) Hvalid expandeds s1 answer Hd_expandeds
         Hargs_expandeds Hkinds_expandeds Hjoin)
      as (e' & -> & Hvalid1 & Hgrows1 & Hk1 & Hkind_e' & Hholds).
    destruct (Hholds s1 (grows_refl s1)) as (Htarget & Hbound & Hkinds').
    cbn [or_unknown_map].
    split; [ exact Hvalid1 | ]. split; [ exact Hgrows1 | ].
    split.
    + intros ty [= <-]. split; [ exact Htarget | ].
      split.
      * intros index t_index Hin. rewrite Hkind_e'. symmetry.
        exact (Hkind_ts index t_index Hin).
      * intros index t_index use statics valuation valuation' v Hin Huse
          Hsatisfied Hextends Hv.
        pose proof (Hlive index t_index statics valuation v Hin Hv) as Hin'.
        destruct (Hto index t_index Hin') as (e & He & Hin_e).
        rewrite <- (grows_joined_env s s1 index Hgrows1), Huse in He.
        destruct (Hexpand use t_index e He) as (Hvalues & _).
        exact
          (Hbound index e use statics valuation valuation' Hin_e Huse
             Hsatisfied Hextends v (Hvalues statics valuation v Hsatisfied Hv)).
    + intros Hk. split; [ exact (Hk1 Hk) | ].
      intros ty [= <-]. exact (Hkinds' Hk).
Qed.

End Body.

(* The join of types meets its contract, whatever bounds the depth of
   its calls to itself. *)
Theorem n_way_join_is_sound (fuel : nat) :
  type_join_sound
    (n_way_join joined_env n_way_join_simples import_type
       join_env_n_way_join_env_extension expand_head
       reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel).
Proof.
  induction fuel as [ | fuel IH].
  - intros s s' ts result _ _ _ _ _ _ Hstep. discriminate.
  - apply (n_way_join_body_is_sound _ IH). intros s.
    destruct fuel; reflexivity.
Qed.

(* The unknown type, answered at the bound on the expansions, meets
   the contract. *)
Lemma unknown_at_bound_is_sound : type_join_sound unknown_at_bound.
Proof.
  intros s s' ts result Hvalid Hwf Hkinded Hdistinct Hscope Hkinds Hstep.
  unfold unknown_at_bound in Hstep.
  destruct ts as [ | [id1 t1] others]; [ discriminate | ].
  cbv zeta in Hstep.
  destruct
    (forallb (fun t => kind_eqb (kind t1) (kind (snd t))) others) eqn:Hall;
    cbn [negb] in Hstep; [ | discriminate ].
  injection Hstep as <- <-.
  assert (Hkind_ts :
    ∀ (index : Index.t) (t : flambda_type),
      In (index, t) ((id1, t1) :: others) -> kind t = kind t1).
  { intros index t [[= <- <-] | Hin]; [ reflexivity | ].
    rewrite forallb_forall in Hall. symmetry.
    exact (kind_eqb_true _ _ (Hall (index, t) Hin)). }
  split; [ exact Hvalid | ]. split; [ apply grows_refl | ]. split.
  - intros ty [= <-]. split.
    { intros x Hx. rewrite free_variables_unknown in Hx. destruct Hx. }
    split.
    { intros index t Hin. rewrite kind_unknown. symmetry.
      exact (Hkind_ts index t Hin). }
    intros index t use statics valuation valuation' v Hin _ _ _ Hv.
    apply unknown_satisfied. rewrite (type_satisfied_kind _ _ _ _ Hv).
    exact (Hkind_ts index t Hin).
  - intros Hk. split; [ exact Hk | ].
    intros ty [= <-] simple kind' Hsimple.
    rewrite kinds_with_unknown in Hsimple. destruct Hsimple.
Qed.

(* The join of types with a bound on its expansions answers nothing
   for no type. *)
Lemma n_way_join_bounded_nil (fuel depth : nat) (s : state) :
  n_way_join_bounded joined_env n_way_join_simples import_type
    join_env_n_way_join_env_extension expand_head
    reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel depth
    s []
  = None.
Proof. destruct fuel; reflexivity. Qed.

(* It meets its contract, whatever the two bounds. *)
Theorem n_way_join_bounded_is_sound (fuel depth : nat) :
  type_join_sound
    (n_way_join_bounded joined_env n_way_join_simples import_type
       join_env_n_way_join_env_extension expand_head
       reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel
       depth).
Proof.
  revert depth. induction fuel as [ | fuel IH]; intros depth.
  - intros s s' ts result _ _ _ _ _ _ Hstep. discriminate.
  - intros s s' ts result Hvalid Hwf Hkinded Hdistinct Hscope Hkinds Hstep.
    cbn [n_way_join_bounded] in Hstep.
    destruct (expands_an_alias ts); [ destruct depth as [ | depth] | ].
    + exact (unknown_at_bound_is_sound s s' ts result Hvalid Hwf Hkinded
               Hdistinct Hscope Hkinds Hstep).
    + exact (n_way_join_body_is_sound _ (IH depth)
               (n_way_join_bounded_nil fuel depth) s s' ts result Hvalid Hwf
               Hkinded Hdistinct Hscope Hkinds Hstep).
    + exact (n_way_join_body_is_sound _ (IH depth)
               (n_way_join_bounded_nil fuel depth) s s' ts result Hvalid Hwf
               Hkinded Hdistinct Hscope Hkinds Hstep).
Qed.

End Functions.

End Traversal.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem flat_heads_sound : flat_heads_sound_statement.
Proof.
  intros typing_env_well_formed state joined_env definitions_of_existentials
    kind_in_target_env state_valid state_kinded n_way_join_simples
    import_type join_env_n_way_join_env_extension
    reduce_head_of_kind_naked_immediate
    (_ & Hsimples & _ & Hextension) Hreduce.
  split; [ apply n_way_join_unknown_is_sound | ].
  split; [ apply n_way_join_relation_simples_is_sound; exact Hsimples | ].
  split; [ apply n_way_join_is_null_is_sound; exact Hsimples | ].
  split;
    [ apply n_way_join_head_of_kind_naked_immediate_is_sound; exact Hreduce
    | ].
  split; [ apply n_way_join_head_of_kind_naked_numbers_is_sound | ].
  split; [ apply n_way_join_env_extension_is_sound; exact Hextension | ].
  split;
    [ exact deduce_is_int_simple_is_sound
    | exact deduce_get_tag_simple_is_sound ].
Qed.

Theorem n_way_join_sound : n_way_join_sound_statement.
Proof.
  intros typing_env_well_formed state joined_env definitions_of_existentials
    kind_in_target_env state_valid state_kinded n_way_join_simples
    import_type join_env_n_way_join_env_extension expand_head
    reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel
    (_ & Hsimples & Himport & Hextension) Hconstants Hexpand Hreduce Heqb
    Hshape.
  refine
    (n_way_join_is_sound _ _ _ _ _ _ _ _ _ _ _ _ _ Hsimples Himport
       Hconstants Hexpand Hreduce Heqb Hshape false false _ _ _ fuel).
  - intros s s' ty Hvalid Hwf Hkinded Himported Hk Hty.
    exact
      (proj2 (proj2 (proj2 (Himport s s' ty Hvalid Hwf Hkinded Himported))) Hk
         Hty).
  - exact
      (n_way_join_env_extension_is_sound typing_env_well_formed joined_env
         definitions_of_existentials kind_in_target_env state_valid
         state_kinded join_env_n_way_join_env_extension Hextension).
  - intros tenv ty expanded He _ Hk Hty.
    exact (proj1 (proj2 (proj2 (Hexpand tenv ty expanded He))) Hk Hty).
Qed.

Theorem n_way_join_bounded_sound : n_way_join_bounded_sound_statement.
Proof.
  intros typing_env_well_formed state joined_env definitions_of_existentials
    kind_in_target_env state_valid state_kinded n_way_join_simples
    import_type join_env_n_way_join_env_extension expand_head
    reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel depth
    (_ & Hsimples & Himport & Hextension) Hconstants Hexpand Hreduce Heqb
    Hshape.
  refine
    (n_way_join_bounded_is_sound _ _ _ _ _ _ _ _ _ _ _ _ _ Hsimples Himport
       Hconstants Hexpand Hreduce Heqb Hshape false false _ _ _ fuel depth).
  - intros s s' ty Hvalid Hwf Hkinded Himported Hk Hty.
    exact
      (proj2 (proj2 (proj2 (Himport s s' ty Hvalid Hwf Hkinded Himported))) Hk
         Hty).
  - exact
      (n_way_join_env_extension_is_sound typing_env_well_formed joined_env
         definitions_of_existentials kind_in_target_env state_valid
         state_kinded join_env_n_way_join_env_extension Hextension).
  - intros tenv ty expanded He _ Hk Hty.
    exact (proj1 (proj2 (proj2 (Hexpand tenv ty expanded He))) Hk Hty).
Qed.

Theorem n_way_join_bound_is_met : n_way_join_bound_is_met_statement.
Proof.
  intros typing_env_well_formed state joined_env definitions_of_existentials
    kind_in_target_env state_valid state_kinded n_way_join_simples
    import_type join_env_n_way_join_env_extension expand_head
    reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel
    Henvironment Hconstants Hexpand Hreduce Heqb Hshape s s' ts ty Hvalid Hwf
    Hkinded Hdistinct Hscope Hkinds Hstep index t_index use statics valuation
    v Hin Huse Hsatisfied Hv.
  destruct
    (n_way_join_sound typing_env_well_formed state joined_env
       definitions_of_existentials kind_in_target_env state_valid
       state_kinded n_way_join_simples import_type
       join_env_n_way_join_env_extension expand_head
       reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel
       Henvironment Hconstants Hexpand Hreduce Heqb Hshape s s' ts _ Hvalid
       Hwf Hkinded Hdistinct Hscope Hkinds Hstep)
    as (Hvalid' & _ & Hknown & _).
  destruct (Hknown ty eq_refl) as (_ & _ & Hbound).
  destruct Henvironment as (Hwitnesses & _).
  destruct (Hwitnesses s' index use statics valuation Hvalid' Huse Hsatisfied)
    as [valuation' Hextends].
  exists valuation'. split; [ exact Hextends | ].
  exact
    (Hbound index t_index use statics valuation valuation' v Hin Huse
       Hsatisfied Hextends Hv).
Qed.

(* The names of the equations inside what the join of types answers:
   the join of types is sound at the lists that have them, for the
   environments that are well formed and have the names of the
   equations inside their types at their kinds. *)
Theorem n_way_join_equations_kinded : n_way_join_equations_kinded_statement.
Proof.
  intros typing_env_well_formed state joined_env definitions_of_existentials
    kind_in_target_env state_valid state_kinded n_way_join_simples
    import_type join_env_n_way_join_env_extension expand_head
    reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel
    (_ & Hsimples & Himport & Hextension) Himport_equations
    Hextension_equations Hconstants Hexpand Hexpand_equations Hreduce Heqb
    Hshape.
  set (wf :=
    fun tenv : Typing_env.t =>
      typing_env_well_formed tenv ∧ types_equations_kinded tenv).
  assert (Hweaken :
    ∀ s : state,
      joined_envs_well_formed wf joined_env s ->
      joined_envs_well_formed typing_env_well_formed joined_env s).
  { intros s H index use Huse. exact (proj1 (H index use Huse)). }
  assert (Hsimples' :
    n_way_join_simples_sound wf joined_env definitions_of_existentials
      kind_in_target_env state_valid state_kinded n_way_join_simples).
  { intros s s' kind simples result Hvalid Hwf.
    exact (Hsimples s s' kind simples result Hvalid (Hweaken s Hwf)). }
  assert (Himport' :
    import_type_sound wf joined_env definitions_of_existentials
      kind_in_target_env state_valid state_kinded import_type).
  { intros s s' ty Hvalid Hwf.
    exact (Himport s s' ty Hvalid (Hweaken s Hwf)). }
  assert (Hextension' :
    n_way_join_env_extension_sound wf joined_env definitions_of_existentials
      kind_in_target_env state_valid state_kinded
      join_env_n_way_join_env_extension).
  { intros s extensions result Hvalid Hwf.
    exact (Hextension s extensions result Hvalid (Hweaken s Hwf)). }
  assert (Hconstants' :
    joined_envs_constants_of_kind wf joined_env state_valid).
  { intros s index use Hvalid Hwf.
    exact (Hconstants s index use Hvalid (Hweaken s Hwf)). }
  assert (Hexpand' : expand_head_sound wf expand_head).
  { intros tenv ty expanded He.
    destruct (Hexpand tenv ty expanded He) as (H1 & H2 & H3 & H4).
    split; [ exact H1 | ]. split; [ intros [Hwf _]; exact (H2 Hwf) | ].
    split; [ exact H3 | exact H4 ]. }
  assert (Hsound :
    type_join_sound wf joined_env definitions_of_existentials
      kind_in_target_env state_valid state_kinded true false
      (n_way_join joined_env n_way_join_simples import_type
         join_env_n_way_join_env_extension expand_head
         reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel)).
  { apply n_way_join_is_sound; try assumption.
    - (* The import of a type. *)
      intros s s' ty Hvalid Hwf Hkinded Himported Hk
        (index & use & Huse & Hty).
      destruct (proj1 (kinds_with_equations_all (simple_of_kind use) ty) Hty)
        as [Haliases Hnames].
      destruct
        (proj2 (proj2 (proj2 (Himport' s s' ty Hvalid Hwf Hkinded Himported)))
           Hk (ex_intro _ index (ex_intro _ use (conj Huse Haliases))))
        as [Hk' Hkinded'].
      split; [ exact Hk' | ].
      refine
        (proj2
           (kinds_with_equations_all
              (simple_of_kind_in_target joined_env kind_in_target_env s') ty)
           (conj Hkinded' _)).
      exact
        (Himport_equations s s' ty Hvalid (Hweaken s Hwf) Hkinded Hk
           (ex_intro _ index (ex_intro _ use (conj Huse Hnames))) Himported).
    - (* The join of extensions. *)
      intros s s' extensions answer Hvalid Hwf Hkinded Hdistinct Hargs Hstep.
      assert (Hargs_aliases :
        join_args_in_scope joined_env
          (arg_in_scope_and_kinded free_variables_of_env_extension
             aliases_of_env_extension)
          s extensions).
      { intros index extension Hin.
        destruct (Hargs index extension Hin) as (use & Huse & Hknown & Hkinds).
        exists use. split; [ exact Huse | ]. split; [ exact Hknown | ].
        exact (proj1 (proj1 (kinds_of_extension_all (simple_of_kind use)
                               extension)
                        Hkinds)). }
      destruct
        (n_way_join_env_extension_is_sound wf joined_env
           definitions_of_existentials kind_in_target_env state_valid
           state_kinded join_env_n_way_join_env_extension Hextension' s s'
           extensions answer Hvalid Hwf Hkinded Hdistinct Hargs_aliases Hstep)
        as (Hvalid' & Hgrows & Htarget & Hbound & Hkinds).
      split; [ exact Hvalid' | ]. split; [ exact Hgrows | ].
      split; [ exact Htarget | ]. split; [ exact Hbound | ].
      intros Hk. destruct (Hkinds Hk) as [Hk' Haliases].
      split; [ exact Hk' | ].
      refine
        (proj2
           (kinds_of_extension_all
              (simple_of_kind_in_target joined_env kind_in_target_env s')
              answer)
           (conj Haliases _)).
      unfold n_way_join_env_extension in Hstep.
      destruct (join_env_n_way_join_env_extension s extensions)
        as [[[joined s1] | ] | ] eqn:Hjoin; simpl in Hstep;
        [ | | discriminate Hstep ]; injection Hstep as <- <-.
      + apply (Hextension_equations s s1 extensions joined Hvalid
                 (Hweaken s Hwf) Hkinded Hk);
          [ | exact Hjoin ].
        intros index extension Hin.
        destruct (Hargs index extension Hin) as (use & Huse & _ & Hkinds').
        exists use. split; [ exact Huse | ].
        exact (proj2 (proj1 (kinds_of_extension_all (simple_of_kind use)
                               extension)
                        Hkinds')).
      + intros n kind [].
    - (* The expansion of a head. *)
      intros tenv ty expanded He [Hwf Hequations] Hk Hty.
      destruct (proj1 (kinds_with_equations_all (simple_of_kind tenv) ty) Hty)
        as [Haliases Hnames].
      refine
        (proj2
           (kinds_with_equations_all (simple_of_kind tenv)
              (Expanded_type.to_type expanded))
           (conj _ _)).
      + exact (proj1 (proj2 (proj2 (Hexpand tenv ty expanded He))) Hk
                 Haliases).
      + exact (Hexpand_equations tenv ty expanded He Hequations Hnames). }
  intros s s' ts ty Hvalid Hwf Hkinded Hequations Hdistinct Hscope Hkinds
    Hnames Hk Hstep.
  assert (Hwf' : joined_envs_well_formed wf joined_env s).
  { intros index use Huse.
    split; [ exact (Hwf index use Huse) | exact (Hequations index use Huse) ]. }
  assert (Hargs :
    join_args_in_scope joined_env (kinded_with true false) s ts).
  { intros index t Hin. destruct (Hkinds index t Hin) as (use & Huse & Ha).
    destruct (Hnames index t Hin) as (use' & Huse' & Hn).
    rewrite Huse in Huse'. injection Huse' as <-. exists use.
    split; [ exact Huse | ].
    exact (proj2 (kinds_with_equations_all (simple_of_kind use) t)
             (conj Ha Hn)). }
  destruct
    (Hsound s s' ts (Or_unknown.Known ty) Hvalid Hwf' Hkinded Hdistinct Hscope
       Hargs Hstep)
    as (_ & _ & _ & Hkinded').
  exact
    (proj2
       (proj1
          (kinds_with_equations_all
             (simple_of_kind_in_target joined_env kind_in_target_env s') ty)
          (proj2 (Hkinded' Hk) ty eq_refl))).
Qed.

(* The type that the join of types answers is well formed: the join of
   types is sound at the lists that have the checks, for the
   environments that are well formed and have well-formed types. *)
Theorem n_way_join_well_formed : n_way_join_well_formed_statement.
Proof.
  intros typing_env_well_formed state joined_env definitions_of_existentials
    kind_in_target_env state_valid state_kinded n_way_join_simples
    import_type join_env_n_way_join_env_extension expand_head
    reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel
    (_ & Hsimples & Himport & Hextension) Hextension_wf Hconstants Hexpand
    Hexpand_wf Hreduce Heqb Hshape.
  set (wf :=
    fun tenv : Typing_env.t =>
      typing_env_well_formed tenv ∧ types_well_formed tenv).
  assert (Hweaken :
    ∀ s : state,
      joined_envs_well_formed wf joined_env s ->
      joined_envs_well_formed typing_env_well_formed joined_env s).
  { intros s H index use Huse. exact (proj1 (H index use Huse)). }
  assert (Hsimples' :
    n_way_join_simples_sound wf joined_env definitions_of_existentials
      kind_in_target_env state_valid state_kinded n_way_join_simples).
  { intros s s' kind simples result Hvalid Hwf.
    exact (Hsimples s s' kind simples result Hvalid (Hweaken s Hwf)). }
  assert (Himport' :
    import_type_sound wf joined_env definitions_of_existentials
      kind_in_target_env state_valid state_kinded import_type).
  { intros s s' ty Hvalid Hwf.
    exact (Himport s s' ty Hvalid (Hweaken s Hwf)). }
  assert (Hextension' :
    n_way_join_env_extension_sound wf joined_env definitions_of_existentials
      kind_in_target_env state_valid state_kinded
      join_env_n_way_join_env_extension).
  { intros s extensions result Hvalid Hwf.
    exact (Hextension s extensions result Hvalid (Hweaken s Hwf)). }
  assert (Hconstants' :
    joined_envs_constants_of_kind wf joined_env state_valid).
  { intros s index use Hvalid Hwf.
    exact (Hconstants s index use Hvalid (Hweaken s Hwf)). }
  assert (Hexpand' : expand_head_sound wf expand_head).
  { intros tenv ty expanded He.
    destruct (Hexpand tenv ty expanded He) as (H1 & H2 & H3 & H4).
    split; [ exact H1 | ]. split; [ intros [Hwf _]; exact (H2 Hwf) | ].
    split; [ exact H3 | exact H4 ]. }
  (* What the lists with the checks say, in an environment and in the
     target. *)
  pose proof
    (fun use : Typing_env.t =>
       kinds_checked (simple_of_kind use) (ill_formed_of_no_kind use) false)
    as Hchecked.
  assert (Hill :
    ∀ s : state,
      ¬ simple_of_kind_in_target joined_env kind_in_target_env s
          (fst ill_formed) (snd ill_formed)).
  { intros s H. cbn in H. discriminate H. }
  pose proof
    (fun s : state =>
       kinds_checked (simple_of_kind_in_target joined_env kind_in_target_env s)
         (Hill s) false)
    as Hchecked_target.
  assert (Hsound :
    type_join_sound wf joined_env definitions_of_existentials
      kind_in_target_env state_valid state_kinded false true
      (n_way_join joined_env n_way_join_simples import_type
         join_env_n_way_join_env_extension expand_head
         reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel)).
  { apply n_way_join_is_sound; try assumption.
    - (* The import of a type: the type is as it was. *)
      intros s s' ty Hvalid Hwf Hkinded Himported Hk
        (index & use & Huse & Hty).
      destruct (proj1 (proj1 (Hchecked use) ty) Hty)
        as [Haliases Hty_wf].
      destruct
        (proj2 (proj2 (proj2 (Himport' s s' ty Hvalid Hwf Hkinded Himported)))
           Hk (ex_intro _ index (ex_intro _ use (conj Huse Haliases))))
        as [Hk' Hkinded'].
      split; [ exact Hk' | ].
      exact (proj2 (proj1 (Hchecked_target s') ty) (conj Hkinded' Hty_wf)).
    - (* The join of extensions. *)
      intros s s' extensions answer Hvalid Hwf Hkinded Hdistinct Hargs Hstep.
      assert (Hargs_aliases :
        join_args_in_scope joined_env
          (arg_in_scope_and_kinded free_variables_of_env_extension
             aliases_of_env_extension)
          s extensions).
      { intros index extension Hin.
        destruct (Hargs index extension Hin) as (use & Huse & Hknown & Hkinds).
        exists use. split; [ exact Huse | ]. split; [ exact Hknown | ].
        exact
          (proj1
             (proj1
                (proj2 (proj2 (proj2 (proj2 (proj2 (proj2 (Hchecked use))))))
                   extension)
                Hkinds)). }
      destruct
        (n_way_join_env_extension_is_sound wf joined_env
           definitions_of_existentials kind_in_target_env state_valid
           state_kinded join_env_n_way_join_env_extension Hextension' s s'
           extensions answer Hvalid Hwf Hkinded Hdistinct Hargs_aliases Hstep)
        as (Hvalid' & Hgrows & Htarget & Hbound & Hkinds).
      split; [ exact Hvalid' | ]. split; [ exact Hgrows | ].
      split; [ exact Htarget | ]. split; [ exact Hbound | ].
      intros Hk. destruct (Hkinds Hk) as [Hk' Haliases].
      split; [ exact Hk' | ].
      refine
        (proj2
           (proj2 (proj2 (proj2 (proj2 (proj2 (proj2 (Hchecked_target s'))))))
              answer)
           (conj Haliases _)).
      unfold n_way_join_env_extension in Hstep.
      destruct (join_env_n_way_join_env_extension s extensions)
        as [[[joined s1] | ] | ] eqn:Hjoin; simpl in Hstep;
        [ | | discriminate Hstep ]; injection Hstep as <- <-.
      + apply (Hextension_wf s s1 extensions joined Hvalid); [ | exact Hjoin ].
        intros index extension Hin.
        destruct (Hargs index extension Hin) as (use & Huse & _ & Hkinds').
        exact
          (proj2
             (proj1
                (proj2 (proj2 (proj2 (proj2 (proj2 (proj2 (Hchecked use))))))
                   extension)
                Hkinds')).
      + reflexivity.
    - (* The expansion of a head. *)
      intros tenv ty expanded He [Hwf Htypes] Hk Hty.
      destruct (proj1 (proj1 (Hchecked tenv) ty) Hty) as [Haliases Hty_wf].
      refine
        (proj2 (proj1 (Hchecked tenv) (Expanded_type.to_type expanded))
           (conj _ _)).
      + exact (proj1 (proj2 (proj2 (Hexpand tenv ty expanded He))) Hk
                 Haliases).
      + exact (Hexpand_wf tenv ty expanded He Htypes Hty_wf). }
  intros s s' ts ty Hvalid Hwf Hkinded Htypes Hdistinct Hscope Hkinds
    Hts_wf Hk Hstep.
  assert (Hwf' : joined_envs_well_formed wf joined_env s).
  { intros index use Huse.
    split; [ exact (Hwf index use Huse) | exact (Htypes index use Huse) ]. }
  assert (Hargs :
    join_args_in_scope joined_env (kinded_with false true) s ts).
  { intros index t Hin. destruct (Hkinds index t Hin) as (use & Huse & Ha).
    exists use. split; [ exact Huse | ].
    exact (proj2 (proj1 (Hchecked use) t) (conj Ha (Hts_wf index t Hin))). }
  destruct
    (Hsound s s' ts (Or_unknown.Known ty) Hvalid Hwf' Hkinded Hdistinct Hscope
       Hargs Hstep)
    as (_ & _ & _ & Hkinded').
  exact
    (proj2
       (proj1 (proj1 (Hchecked_target s') ty)
          (proj2 (Hkinded' Hk) ty eq_refl))).
Qed.
